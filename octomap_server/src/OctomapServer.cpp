/*
 * Copyright (c) 2010-2013, A. Hornung, University of Freiburg
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of Freiburg nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <octomap_server/OctomapServer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <fstream>
#include "octomap/octomap.h"
#include "octomap/Pointcloud.h"
#include "octomap/octomap_types.h"
#include "octomap/math/Vector3.h"
#include <iostream>
 #include <unistd.h>

using namespace octomap;
using namespace cv;
using namespace std;
using namespace octomap_msgs;

int num;
int flag;

namespace octomap_server{

OctomapServer::OctomapServer(ros::NodeHandle private_nh_)
: m_nh(),
  m_pointCloudSub(NULL),
  m_rgbSub(NULL),
  m_depSub(NULL),
  m_infoSub(NULL),
  m_tfPointCloudSub(NULL),
  m_octree(NULL),
  m_maxRange(-1.0),
  m_worldFrameId("robot/world"), m_baseFrameId("base_footprint"),
  m_useHeightMap(true),
  m_useColoredMap(false),
  m_colorFactor(0.8),
  m_latchedTopics(true),
  m_publishFreeSpace(false),
  m_res(0.02),
  m_treeDepth(0),
  m_maxTreeDepth(0),
  m_pointcloudMinZ(-std::numeric_limits<double>::max()),
  m_pointcloudMaxZ(std::numeric_limits<double>::max()),
  m_occupancyMinZ(-std::numeric_limits<double>::max()),
  m_occupancyMaxZ(std::numeric_limits<double>::max()),
  m_minSizeX(0.0), m_minSizeY(0.0),
  m_filterSpeckles(false), m_filterGroundPlane(false),
  m_groundFilterDistance(0.04), m_groundFilterAngle(0.15), m_groundFilterPlaneDistance(0.07),
  m_compressMap(true),
  m_incrementalUpdate(false),
  m_robot_h(0.2),
  m_height(1),
  m_dynamicmap(true),
  m_completemap(false),
  robot_id(0),
  model(0),
  loop(false),
  box(0)
{
  double probHit, probMiss, thresMin, thresMax;

  ros::NodeHandle private_nh(private_nh_);
  
  private_nh.param("robot_id",robot_id,robot_id);	
  private_nh.param("dynamic_map",m_dynamicmap,m_dynamicmap);
  private_nh.param("complete_map",m_completemap,m_completemap);	
  private_nh.param("robot_h",m_robot_h,m_robot_h);	
  private_nh.param("height",m_height,m_height);
  private_nh.param("model",model,model);
  private_nh.param("box",box,box);	

  private_nh.param("frame_id", m_worldFrameId, m_worldFrameId);
  private_nh.param("base_frame_id", m_baseFrameId, m_baseFrameId);
  private_nh.param("height_map", m_useHeightMap, m_useHeightMap);
  private_nh.param("colored_map", m_useColoredMap, m_useColoredMap);
  private_nh.param("color_factor", m_colorFactor, m_colorFactor);

  private_nh.param("pointcloud_min_z", m_pointcloudMinZ,m_pointcloudMinZ);
  private_nh.param("pointcloud_max_z", m_pointcloudMaxZ,m_pointcloudMaxZ);
  private_nh.param("occupancy_min_z", m_occupancyMinZ,m_occupancyMinZ);
  private_nh.param("occupancy_max_z", m_occupancyMaxZ,m_occupancyMaxZ);
  private_nh.param("min_x_size", m_minSizeX,m_minSizeX);
  private_nh.param("min_y_size", m_minSizeY,m_minSizeY);

  private_nh.param("filter_speckles", m_filterSpeckles, m_filterSpeckles);
  private_nh.param("filter_ground", m_filterGroundPlane, m_filterGroundPlane);
  // distance of points from plane for RANSAC
  private_nh.param("ground_filter/distance", m_groundFilterDistance, m_groundFilterDistance);
  // angular derivation of found plane:
  private_nh.param("ground_filter/angle", m_groundFilterAngle, m_groundFilterAngle);
  // distance of found plane from z=0 to be detected as ground (e.g. to exclude tables)
  private_nh.param("ground_filter/plane_distance", m_groundFilterPlaneDistance, m_groundFilterPlaneDistance);

  private_nh.param("sensor_model/max_range", m_maxRange, m_maxRange);

  private_nh.param("resolution", m_res, m_res);
  private_nh.param("sensor_model/hit", probHit, 0.7);
  private_nh.param("sensor_model/miss", probMiss, 0.4);
  private_nh.param("sensor_model/min", thresMin, 0.12);
  private_nh.param("sensor_model/max", thresMax, 0.97);
  private_nh.param("compress_map", m_compressMap, m_compressMap);
  private_nh.param("incremental_2D_projection", m_incrementalUpdate, m_incrementalUpdate);

  if (m_filterGroundPlane && (m_pointcloudMinZ > 0.0 || m_pointcloudMaxZ < 0.0)){
    ROS_WARN_STREAM("You enabled ground filtering but incoming pointclouds will be pre-filtered in ["
              <<m_pointcloudMinZ <<", "<< m_pointcloudMaxZ << "], excluding the ground level z=0. "
              << "This will not work.");
  }

  if (m_useHeightMap && m_useColoredMap) {
    ROS_WARN_STREAM("You enabled both height map and RGB color registration. This is contradictory. Defaulting to height map.");
    m_useColoredMap = false;
  }

  if (m_useColoredMap) {
#ifdef COLOR_OCTOMAP_SERVER
    ROS_INFO_STREAM("Using RGB color registration (if information available)");
#else
    ROS_ERROR_STREAM("Colored map requested in launch file - node not running/compiled to support colors, please define COLOR_OCTOMAP_SERVER and recompile or launch the octomap_color_server node");
#endif
  }


  // initialize octomap object & params
  ROS_ERROR("res %f",m_res);
  m_octree = new OcTreeT(m_res);
  m_octree->setProbHit(probHit);
  m_octree->setProbMiss(probMiss);
  m_octree->setClampingThresMin(thresMin);
  m_octree->setClampingThresMax(thresMax);
/*  m_octree->setProbHit(2.71828);
  m_octree->setProbMiss(0.36787);
  m_octree->setClampingThresMin(0.1192);
  m_octree->setClampingThresMax(0.8807);
  */
  m_treeDepth = m_octree->getTreeDepth();
  m_maxTreeDepth = m_treeDepth;
  m_gridmap.info.resolution = m_res;

	num=0;
	flag=0;

  double r, g, b, a;
  private_nh.param("color/r", r, 0.0);
  private_nh.param("color/g", g, 0.0);
  private_nh.param("color/b", b, 1.0);
  private_nh.param("color/a", a, 1.0);
  m_color.r = r;
  m_color.g = g;
  m_color.b = b;
  m_color.a = a;

  private_nh.param("color_free/r", r, 0.0);
  private_nh.param("color_free/g", g, 1.0);
  private_nh.param("color_free/b", b, 0.0);
  private_nh.param("color_free/a", a, 1.0);
  m_colorFree.r = r;
  m_colorFree.g = g;
  m_colorFree.b = b;
  m_colorFree.a = a;
  anend[0]=anend[1]=anend[2]=0;

  private_nh.param("publish_free_space", m_publishFreeSpace, m_publishFreeSpace);

  private_nh.param("latch", m_latchedTopics, m_latchedTopics);
  if (m_latchedTopics){
    ROS_INFO("Publishing latched (single publish will take longer, all topics are prepared)");
  } else
    ROS_INFO("Publishing non-latched (topics are only prepared as needed, will only be re-published on map change");

  m_markerPub = m_nh.advertise<visualization_msgs::MarkerArray>("occupied_cells_vis_array", 1, m_latchedTopics);
  m_binaryMapPub = m_nh.advertise<Octomap>("octomap_binary", 1, m_latchedTopics);
  m_fullMapPub = m_nh.advertise<Octomap>("octomap_full", 1, m_latchedTopics);
  m_pointCloudPub = m_nh.advertise<sensor_msgs::PointCloud2>("octomap_point_cloud_centers", 1, m_latchedTopics);
  m_mapPub = m_nh.advertise<nav_msgs::OccupancyGrid>("projected_map", 5);
  m_fmarkerPub = m_nh.advertise<visualization_msgs::MarkerArray>("free_cells_vis_array", 1, m_latchedTopics);


  //add by lucky
  cmdpub=m_nh.advertise<std_msgs::String>("/tmp/cmd",1);
  cmdsub=m_nh.subscribe<std_msgs::String>("/tmp/cmd",1,&OctomapServer::loopcmd,this);
  pospub=m_nh.advertise<octomap_msgs::Pos>("robot_pos",5);
  octreepub=m_nh.advertise<octomap_msgs::Octomap>("/merge/octree",10);
  getfile=m_nh.subscribe("/octomap/cmd",10,&OctomapServer::savemap,this);
  setanend=m_nh.subscribe("/map/end",10,&OctomapServer::setbyhuman,this);

  if(model==0)
  {
	  rgbpub=m_nh.advertise<sensor_msgs::Image>("/octomap/tmp/rgb",10);
	  deppub=m_nh.advertise<sensor_msgs::Image>("/octomap/tmp/depth",10);
	  infopub=m_nh.advertise<sensor_msgs::CameraInfo>("/octomap/tmp/info",10);
	  m_pointCloudSub = new message_filters::Subscriber<sensor_msgs::PointCloud2> (m_nh, "/tmp/pointcloud", 1);
	  m_rgbSub = new message_filters::Subscriber<sensor_msgs::Image> (m_nh, "/tmp/rgb", 1);
	  m_depSub = new message_filters::Subscriber<sensor_msgs::Image> (m_nh, "/tmp/depth", 1);
	  m_infoSub = new message_filters::Subscriber<sensor_msgs::CameraInfo> (m_nh, "/tmp/info", 1);

	  sync_ = new message_filters::Synchronizer<Sync>(Sync(1),  *m_rgbSub, *m_depSub, *m_infoSub, *m_pointCloudSub);
	  sync_->registerCallback(boost::bind(&OctomapServer::msgcallback,this,_1,_2,_3,_4));
  }
  else
  {

	  m_pointCloudSub = new message_filters::Subscriber<sensor_msgs::PointCloud2> (m_nh, "/tmp/pointcloud", 1);
	  m_tfPointCloudSub = new tf::MessageFilter<sensor_msgs::PointCloud2> (*m_pointCloudSub, m_tfListener, m_worldFrameId, 1);
	  m_tfPointCloudSub->registerCallback(boost::bind(&OctomapServer::insertCloudCallback, this, _1));
  }


  m_octomapBinaryService = m_nh.advertiseService("octomap_binary", &OctomapServer::octomapBinarySrv, this);
  m_octomapFullService = m_nh.advertiseService("octomap_full", &OctomapServer::octomapFullSrv, this);
  m_clearBBXService = private_nh.advertiseService("clear_bbx", &OctomapServer::clearBBXSrv, this);
  m_resetService = private_nh.advertiseService("reset", &OctomapServer::resetSrv, this);

  dynamic_reconfigure::Server<OctomapServerConfig>::CallbackType f;

  f = boost::bind(&OctomapServer::reconfigureCallback, this, _1, _2);
  m_reconfigureServer.setCallback(f);
}

OctomapServer::~OctomapServer(){
	if (m_tfPointCloudSub){
		delete m_tfPointCloudSub;
		m_tfPointCloudSub = NULL;
	}

	if (m_pointCloudSub){
		delete m_pointCloudSub;
		m_pointCloudSub = NULL;
	}


	if (m_octree){
		delete m_octree;
		m_octree = NULL;
	}

}

bool OctomapServer::openFile(const std::string& filename){
	if (filename.length() <= 3)
		return false;

	std::string suffix = filename.substr(filename.length()-3, 3);
	if (suffix== ".bt"){
		if (!m_octree->readBinary(filename)){
			return false;
		}
	} else if (suffix == ".ot"){
		AbstractOcTree* tree = AbstractOcTree::read(filename);
		if (!tree){
			return false;
		}
		if (m_octree){
			delete m_octree;
			m_octree = NULL;
		}
		m_octree = dynamic_cast<OcTreeT*>(tree);
		if (!m_octree){
			ROS_ERROR("Could not read OcTree in file, currently there are no other types supported in .ot");
			return false;
		}

	} else{
		return false;
	}

	ROS_INFO("Octomap file %s loaded (%zu nodes).", filename.c_str(),m_octree->size());

	m_treeDepth = m_octree->getTreeDepth();
	m_maxTreeDepth = m_treeDepth;
	m_res = m_octree->getResolution();
	m_gridmap.info.resolution = m_res;
	double minX, minY, minZ;
	double maxX, maxY, maxZ;
	m_octree->getMetricMin(minX, minY, minZ);
	m_octree->getMetricMax(maxX, maxY, maxZ);

	m_updateBBXMin[0] = m_octree->coordToKey(minX);
	m_updateBBXMin[1] = m_octree->coordToKey(minY);
	m_updateBBXMin[2] = m_octree->coordToKey(minZ);

	m_updateBBXMax[0] = m_octree->coordToKey(maxX);
	m_updateBBXMax[1] = m_octree->coordToKey(maxY);
	m_updateBBXMax[2] = m_octree->coordToKey(maxZ);

	publishAll();

	return true;

}

void OctomapServer::insertCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud){
	ros::WallTime startTime = ros::WallTime::now();

	//
	// ground filtering in base frame
	//
	if(num==0)
	{
		m_octree->totalexpand();
		num=1;
	}
	tm1.start();
	PCLPointCloud pc; // input cloud for filtering and ground-detection
	pcl::fromROSMsg(*cloud, pc);

	tf::StampedTransform sensorToWorldTf;
	try {
		m_tfListener.lookupTransform(m_worldFrameId, cloud->header.frame_id, cloud->header.stamp, sensorToWorldTf);
	} catch(tf::TransformException& ex){
		ROS_ERROR_STREAM( "Transform error of sensor data: " << ex.what() << ", quitting callback");
		return;
	}

	yaw=tf::getYaw(sensorToWorldTf.getRotation());

	Eigen::Matrix4f sensorToWorld;
	pcl_ros::transformAsMatrix(sensorToWorldTf, sensorToWorld);

	// set up filter for height range, also removes NANs:

	pcl::StatisticalOutlierRemoval<PCLPoint> sor;
	sor.setMeanK(40);
	sor.setStddevMulThresh(1.0);

	pcl::VoxelGrid<PCLPoint> vg;
	vg.setLeafSize(m_res/1.5,m_res/1.5,m_res/1.5);

	PCLPointCloud pc_ground; // segmented ground plane
	PCLPointCloud pc_nonground; // everything else

	TimeTracker tt;
	tt.start();
	if (m_filterGroundPlane){
		tf::StampedTransform sensorToBaseTf, baseToWorldTf;
		try{
			m_tfListener.waitForTransform(m_baseFrameId, cloud->header.frame_id, cloud->header.stamp, ros::Duration(0.2));
			m_tfListener.lookupTransform(m_baseFrameId, cloud->header.frame_id, cloud->header.stamp, sensorToBaseTf);
			m_tfListener.lookupTransform(m_worldFrameId, m_baseFrameId, cloud->header.stamp, baseToWorldTf);


		}catch(tf::TransformException& ex){
			ROS_ERROR_STREAM( "Transform error for ground plane filter: " << ex.what() << ", quitting callback.\n"
					"You need to set the base_frame_id or disable filter_ground.");
		}


		Eigen::Matrix4f sensorToBase, baseToWorld;
		pcl_ros::transformAsMatrix(sensorToBaseTf, sensorToBase);
		pcl_ros::transformAsMatrix(baseToWorldTf, baseToWorld);

		// transform pointcloud from sensor frame to fixed robot frame
		pcl::transformPointCloud(pc, pc, sensorToBase);
		filterGroundPlane(pc, pc_ground, pc_nonground);

		// transform clouds to world frame for insertion
		pcl::transformPointCloud(pc_ground, pc_ground, baseToWorld);
		pcl::transformPointCloud(pc_nonground, pc_nonground, baseToWorld);
	} else {
		// directly transform to map frame:
		pcl::transformPointCloud(pc, pc, sensorToWorld);

		// just filter height range:
		vg.setInputCloud(pc.makeShared());
		vg.filter(pc);

		sor.setInputCloud(pc.makeShared());
		sor.filter(pc);

		pc_nonground = pc;
		// pc_nonground is empty without ground segmentation
		pc_ground.header = pc.header;
		pc_nonground.header = pc.header;
	}
	tt.stop();
	ROS_ERROR("pcl time %d",tt.duration());
	tt.start();
	insertScan(sensorToWorldTf.getOrigin(), pc_ground, pc_nonground);
	tt.stop();
	ROS_ERROR("insert time %d",tt.duration());
	double total_elapsed = (ros::WallTime::now() - startTime).toSec();
	ROS_INFO("Pointcloud insertion in OctomapServer done (%zu+%zu pts (ground/nonground), %f sec)", pc_ground.size(), pc_nonground.size(), total_elapsed);

	tt.start();
	publishAll(cloud->header.stamp);
	tt.stop();
	ROS_ERROR("publish time %d",tt.duration());
	tm1.stop();
	ROS_ERROR("compute time %d\n",tm1.duration());

}
void OctomapServer::msgcallback(
		const	sensor_msgs::ImageConstPtr& rgb,
		const	sensor_msgs::ImageConstPtr& dep,
		const	sensor_msgs::CameraInfoConstPtr& info,
		const	sensor_msgs::PointCloud2ConstPtr& cloud)
{
	ros::WallTime startTime = ros::WallTime::now();

	//
	// ground filtering in base frame
	//
	if(num==0)
	{
		m_octree->totalexpand();
	}
	tm1.start();
	PCLPointCloud pc; // input cloud for filtering and ground-detection
	pcl::fromROSMsg(*cloud, pc);

	tf::StampedTransform sensorToWorldTf;
	try {
		m_tfListener.lookupTransform(m_worldFrameId, cloud->header.frame_id, cloud->header.stamp, sensorToWorldTf);
	} catch(tf::TransformException& ex){
		ROS_ERROR_STREAM( "Transform error of sensor data: " << ex.what() << ", quitting callback");
		return;
	}


	yaw=tf::getYaw(sensorToWorldTf.getRotation());
	rgbpub.publish(rgb);
	deppub.publish(dep);
	infopub.publish(info);
	br.sendTransform(sensorToWorldTf); 

	Eigen::Matrix4f sensorToWorld;
	pcl_ros::transformAsMatrix(sensorToWorldTf, sensorToWorld);

	// set up filter for height range, also removes NANs:

	pcl::VoxelGrid<PCLPoint> vg;
	vg.setLeafSize(m_res-0.005,m_res-0.005,m_res-0.005);

	PCLPointCloud pc_ground; // segmented ground plane
	PCLPointCloud pc_nonground; // everything else

	if (m_filterGroundPlane){
		tf::StampedTransform sensorToBaseTf, baseToWorldTf;
		try{
			m_tfListener.waitForTransform(m_baseFrameId, cloud->header.frame_id, cloud->header.stamp, ros::Duration(0.2));
			m_tfListener.lookupTransform(m_baseFrameId, cloud->header.frame_id, cloud->header.stamp, sensorToBaseTf);
			m_tfListener.lookupTransform(m_worldFrameId, m_baseFrameId, cloud->header.stamp, baseToWorldTf);


		}catch(tf::TransformException& ex){
			ROS_ERROR_STREAM( "Transform error for ground plane filter: " << ex.what() << ", quitting callback.\n"
					"You need to set the base_frame_id or disable filter_ground.");
		}


		Eigen::Matrix4f sensorToBase, baseToWorld;
		pcl_ros::transformAsMatrix(sensorToBaseTf, sensorToBase);
		pcl_ros::transformAsMatrix(baseToWorldTf, baseToWorld);

		// transform pointcloud from sensor frame to fixed robot frame
		pcl::transformPointCloud(pc, pc, sensorToBase);
		filterGroundPlane(pc, pc_ground, pc_nonground);

		// transform clouds to world frame for insertion
		pcl::transformPointCloud(pc_ground, pc_ground, baseToWorld);
		pcl::transformPointCloud(pc_nonground, pc_nonground, baseToWorld);
	} else {
		// directly transform to map frame:
		pcl::transformPointCloud(pc, pc, sensorToWorld);

		// just filter height range:
		vg.setInputCloud(pc.makeShared());
		vg.filter(pc);

		pc_nonground = pc;
		// pc_nonground is empty without ground segmentation
		pc_ground.header = pc.header;
		pc_nonground.header = pc.header;
	}

	TimeTracker tt;
	tt.start();
	insertScan(sensorToWorldTf.getOrigin(), pc_ground, pc_nonground);
	tt.stop();
	ROS_ERROR("insert time %d",tt.duration());
	double total_elapsed = (ros::WallTime::now() - startTime).toSec();
	ROS_INFO("Pointcloud insertion in OctomapServer done (%zu+%zu pts (ground/nonground), %f sec)", pc_ground.size(), pc_nonground.size(), total_elapsed);

	tt.start();
	publishAll(cloud->header.stamp);
	tt.stop();
	ROS_ERROR("publish time %d",tt.duration());
	tm1.stop();
	ROS_ERROR("compute time %d\n",tm1.duration());
}

void OctomapServer::insertScan(const tf::Point& sensorOriginTf, const PCLPointCloud& ground, const PCLPointCloud& nonground){
	sensorOrigin = pointTfToOctomap(sensorOriginTf);
	if (!m_octree->coordToKeyChecked(sensorOrigin, m_updateBBXMin)
			|| !m_octree->coordToKeyChecked(sensorOrigin, m_updateBBXMax))
	{
		ROS_ERROR_STREAM("Could not generate Key for origin "<<sensorOrigin);
	}

#ifdef COLOR_OCTOMAP_SERVER
	unsigned char* colors = new unsigned char[3];
#endif
	octomap::Pointcloud opcl;
	opcl.reserve(nonground.size()/32);
	for(PCLPointCloud::const_iterator it=nonground.begin();it!=nonground.end();++it)
	{
		float x,y,z,tmpmin,tmpmax;
		x=it->x;
		y=it->y;
		z=it->z;
		if(!std::isnan(x)&&!std::isnan(y)&&!std::isnan(z))
		{
			opcl.push_back(x,y,z);
		}
	}
	m_octree->insertPointCloudfor2d(opcl,sensorOrigin,5.0,m_height+m_robot_h,m_robot_h,false);
//	if (m_compressMap)
//		m_octree->prune();

}



void OctomapServer::publishAll(const ros::Time& rostime){
	ros::WallTime startTime = ros::WallTime::now();
	size_t octomapSize = m_octree->size();
	// TODO: estimate num occ. voxels for size of arrays (reserve)
	if (octomapSize <= 1){
		ROS_WARN("Nothing to publish, octree is empty");
		return;
	}
	bool publishFreeMarkerArray = m_publishFreeSpace && (m_latchedTopics || m_fmarkerPub.getNumSubscribers() > 0);
	bool publishMarkerArray = (m_latchedTopics || m_markerPub.getNumSubscribers() > 0);
	bool publishPointCloud = (m_latchedTopics || m_pointCloudPub.getNumSubscribers() > 0);
	bool publishBinaryMap = (m_latchedTopics || m_binaryMapPub.getNumSubscribers() > 0);
	bool publishFullMap = (m_latchedTopics || m_fullMapPub.getNumSubscribers() > 0);
	m_publish2DMap = (m_latchedTopics || m_mapPub.getNumSubscribers() > 0);
	publishMarkerArray=true;
	// init markers for free space:
	visualization_msgs::MarkerArray freeNodesVis;
	// each array stores all cubes of a different size, one for each depth level:
	freeNodesVis.markers.resize(m_treeDepth+1);

	geometry_msgs::Pose pose;
	pose.orientation = tf::createQuaternionMsgFromYaw(0.0);

	// init markers:
	visualization_msgs::MarkerArray occupiedNodesVis;
	// each array stores all cubes of a different size, one for each depth level:
	occupiedNodesVis.markers.resize(m_treeDepth+1);

	// init pointcloud:
	pcl::PointCloud<PCLPoint> pclCloud;
	// call pre-traversal hook:
	handlePreNodeTraversal(rostime);
	// now, traverse all leafs in the tree:
	if(m_dynamicmap)
	{
		if(!m_completemap)
		{
			int upmaxx,upmaxy,upminx,upminy;
			m_octree->getjustmaxmin(upmaxx,upmaxy,upminx,upminy);
			int limit=pow(2,16)-1;
			clearupdatemap(upmaxx,upmaxy,upminx,upminy);
			upminx=upminx-box;
			upminx=upminx>0?upminx:0;
			upminy=upminy-box;
			upminy=upminy>0?upminy:0;
			upmaxx=upmaxx+box;
			upmaxx=upmaxx<limit?upmaxx:limit;
			upmaxy=upmaxy+box;
			upmaxy=upmaxy<limit?upmaxy:limit;
	
			int smallmin[]={upminx,upminy};
			int smallmax[]={upmaxx,upmaxy};
			octomap::upKey updateNode=m_octree->searchkey(smallmin,smallmax);
			octomap::upKey::iterator upit;	
			for(upit=updateNode.begin();upit!=updateNode.end();upit++)
			{
				if (m_publish2DMap && m_projectCompleteMap){
					update2DMap(upit,upit->occupied);
				}

			}
		}
		else
		{
			double minX, minY, minZ, maxX, maxY, maxZ;
			if(m_useHeightMap)
				m_octree->getMetricMaxMin(maxX, maxY, maxZ, minX, minY, minZ);
			for (OcTreeT::iterator it = m_octree->begin(m_maxTreeDepth),
					end = m_octree->end(); it != end; ++it)
			{
				if (m_octree->isNodeOccupied(*it)){
					double z = it.getZ();
					if (z > -0.75 && z < m_occupancyMaxZ)
					{
						double size = it.getSize();
						double x = it.getX();
						double y = it.getY();

						// Ignore speckles in the map:
						if (m_filterSpeckles && (it.getDepth() == m_treeDepth +1) && isSpeckleNode(it.getKey())){
							ROS_DEBUG("Ignoring single speckle at (%f,%f,%f)", x, y, z);
							continue;
						} // else: current octree node is no speckle, send it out
						handleOccupiedNode(it);
						//create marker:
						if (publishMarkerArray){
							unsigned idx = it.getDepth();
							assert(idx < occupiedNodesVis.markers.size());

							geometry_msgs::Point cubeCenter;
							cubeCenter.x = x;
							cubeCenter.y = y;
							cubeCenter.z = z;
							occupiedNodesVis.markers[idx].points.push_back(cubeCenter);
							if (m_useHeightMap){
								double h = (1.0 - std::min(std::max((cubeCenter.z-minZ)/ (maxZ - minZ), 0.0), 1.0)) *m_colorFactor;
								occupiedNodesVis.markers[idx].colors.push_back(heightMapColor(h));
							}

						}

						// insert into pointcloud:
						/*	  if (publishPointCloud) {
							  pclCloud.push_back(PCLPoint(x, y, z));
							  }*/

					}
				} 
				else{ // node not occupied => mark as free in 2D map if unknown so far
					double z = it.getZ();
					if (z > m_occupancyMinZ && z < m_occupancyMaxZ)
					{
						handleFreeNode(it);
					}
				}
			}
		}
	}
	else
	{

	}

	// call post-traversal hook:
	handlePostNodeTraversal(rostime);


	//	printf("maxx %f maxy %f maxz %f\n",maxx,maxy,maxz);
	//	printf("minx %f miny %f minz %f\n",minx,miny,minz);
	// finish MarkerArray:


	if(m_completemap)
	{	
	   if (publishMarkerArray){
	   for (unsigned i= 0; i < occupiedNodesVis.markers.size(); ++i){
	   double size = m_octree->getNodeSize(i);

	   occupiedNodesVis.markers[i].header.frame_id = m_worldFrameId;
	   occupiedNodesVis.markers[i].header.stamp = rostime;
	   occupiedNodesVis.markers[i].ns = "map";
	   occupiedNodesVis.markers[i].id = i;
	   occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
	   occupiedNodesVis.markers[i].scale.x = size;
	   occupiedNodesVis.markers[i].scale.y = size;
	   occupiedNodesVis.markers[i].scale.z = size;
	   if (!m_useColoredMap)
	   occupiedNodesVis.markers[i].color = m_color;


	   if (occupiedNodesVis.markers[i].points.size() > 0)
	   occupiedNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
	   else
	   occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
	   }

	   m_markerPub.publish(occupiedNodesVis);
	   }
	}
	/*
	// finish FreeMarkerArray:
	if (publishFreeMarkerArray){
	for (unsigned i= 0; i < freeNodesVis.markers.size(); ++i){
	double size = m_octree->getNodeSize(i);

	freeNodesVis.markers[i].header.frame_id = m_worldFrameId;
	freeNodesVis.markers[i].header.stamp = rostime;
	freeNodesVis.markers[i].ns = "map";
	freeNodesVis.markers[i].id = i;
	freeNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
	freeNodesVis.markers[i].scale.x = size;
	freeNodesVis.markers[i].scale.y = size;
	freeNodesVis.markers[i].scale.z = size;
	freeNodesVis.markers[i].color = m_colorFree;


	if (freeNodesVis.markers[i].points.size() > 0)
	freeNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
	else
	freeNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
	}

	m_fmarkerPub.publish(freeNodesVis);
	}


	// finish pointcloud:
	if (publishPointCloud){
	sensor_msgs::PointCloud2 cloud;
	pcl::toROSMsg (pclCloud, cloud);
	cloud.header.frame_id = m_worldFrameId;
	cloud.header.stamp = rostime;
	m_pointCloudPub.publish(cloud);
	}

	if (publishBinaryMap)
	publishBinaryOctoMap(rostime);

	if (publishFullMap)
	publishFullOctoMap(rostime);
	 */
	if(loop)
	{
		std_msgs::String msg;
		string s="OK";
		msg.data=s.c_str();
		cmdpub.publish(msg);
	}
//	double total_elapsed = (ros::WallTime::now() - startTime).toSec();
//	ROS_ERROR("Map publishing in OctomapServer took %f sec", total_elapsed);

}

void OctomapServer::loopcmd(const std_msgs::String::ConstPtr& msg)
{
	string s=msg->data;
	if(s=="loop start")
	{
		loop=true;
		num=0;
		m_octree->clear();
		m_gridmap.data.clear();
		m_gridmap.info.height = 0.0;
		m_gridmap.info.width = 0.0;
		m_gridmap.info.origin.position.x = 0.0;
		m_gridmap.info.origin.position.y = 0.0;
		std_msgs::String msg;
		string s="OK";
		msg.data=s.c_str();
		cmdpub.publish(msg);
	}
	if(s=="loop stop")
		loop=false;
}


bool OctomapServer::octomapBinarySrv(OctomapSrv::Request  &req,
		OctomapSrv::Response &res)
{
	ros::WallTime startTime = ros::WallTime::now();
	ROS_INFO("Sending binary map data on service request");
	res.map.header.frame_id = m_worldFrameId;
	res.map.header.stamp = ros::Time::now();
	if (!octomap_msgs::binaryMapToMsg(*m_octree, res.map))
		return false;

	double total_elapsed = (ros::WallTime::now() - startTime).toSec();
	ROS_ERROR("Binary octomap sent in %f sec", total_elapsed);
	return true;
}

bool OctomapServer::octomapFullSrv(OctomapSrv::Request  &req,
		OctomapSrv::Response &res)
{
	ROS_INFO("Sending full map data on service request");
	res.map.header.frame_id = m_worldFrameId;
	res.map.header.stamp = ros::Time::now();


	if (!octomap_msgs::fullMapToMsg(*m_octree, res.map))
		return false;

	return true;
}

bool OctomapServer::clearBBXSrv(BBXSrv::Request& req, BBXSrv::Response& resp){
	point3d min = pointMsgToOctomap(req.min);
	point3d max = pointMsgToOctomap(req.max);

	double thresMin = m_octree->getClampingThresMin();
	for(OcTreeT::leaf_bbx_iterator it = m_octree->begin_leafs_bbx(min,max),
			end=m_octree->end_leafs_bbx(); it!= end; ++it){

		it->setLogOdds(octomap::logodds(thresMin));
		//			m_octree->updateNode(it.getKey(), -6.0f);
	}
	// TODO: eval which is faster (setLogOdds+updateInner or updateNode)
	//  m_octree->updateInnerOccupancy();

	publishAll(ros::Time::now());

	return true;
}

bool OctomapServer::resetSrv(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp) {
	visualization_msgs::MarkerArray occupiedNodesVis;
	occupiedNodesVis.markers.resize(m_treeDepth +1);
	ros::Time rostime = ros::Time::now();
	m_octree->clear();
	// clear 2D map:
	m_gridmap.data.clear();
	m_gridmap.info.height = 0.0;
	m_gridmap.info.width = 0.0;
	m_gridmap.info.resolution = 0.0;
	m_gridmap.info.origin.position.x = 0.0;
	m_gridmap.info.origin.position.y = 0.0;

	ROS_INFO("Cleared octomap");
	publishAll(rostime);

	publishBinaryOctoMap(rostime);
	for (unsigned i= 0; i < occupiedNodesVis.markers.size(); ++i){

		occupiedNodesVis.markers[i].header.frame_id = m_worldFrameId;
		occupiedNodesVis.markers[i].header.stamp = rostime;
		occupiedNodesVis.markers[i].ns = "map";
		occupiedNodesVis.markers[i].id = i;
		occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
		occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
	}

	m_markerPub.publish(occupiedNodesVis);

	visualization_msgs::MarkerArray freeNodesVis;
	freeNodesVis.markers.resize(m_treeDepth +1);

	for (unsigned i= 0; i < freeNodesVis.markers.size(); ++i){

		freeNodesVis.markers[i].header.frame_id = m_worldFrameId;
		freeNodesVis.markers[i].header.stamp = rostime;
		freeNodesVis.markers[i].ns = "map";
		freeNodesVis.markers[i].id = i;
		freeNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
		freeNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
	}
	m_fmarkerPub.publish(freeNodesVis);

	return true;
}

void OctomapServer::publishBinaryOctoMap(const ros::Time& rostime) const{

	Octomap map;
	map.header.frame_id = m_worldFrameId;
	map.header.stamp = rostime;

	if (octomap_msgs::binaryMapToMsg(*m_octree, map))
		m_binaryMapPub.publish(map);
	else
		ROS_ERROR("Error serializing OctoMap");
}

void OctomapServer::publishFullOctoMap(const ros::Time& rostime) const{

	Octomap map;
	map.header.frame_id = m_worldFrameId;
	map.header.stamp = rostime;

	if (octomap_msgs::fullMapToMsg(*m_octree, map))
		m_fullMapPub.publish(map);
	else
		ROS_ERROR("Error serializing OctoMap");

}


void OctomapServer::filterGroundPlane(const PCLPointCloud& pc, PCLPointCloud& ground, PCLPointCloud& nonground) const{
	ground.header = pc.header;
	nonground.header = pc.header;

	if (pc.size() < 50){
		ROS_WARN("Pointcloud in OctomapServer too small, skipping ground plane extraction");
		nonground = pc;
	} else {
		// plane detection for ground plane removal:
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

		// Create the segmentation object and set up:
		pcl::SACSegmentation<PCLPoint> seg;
		seg.setOptimizeCoefficients (true);
		// TODO: maybe a filtering based on the surface normals might be more robust / accurate?
		seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setMaxIterations(200);
		seg.setDistanceThreshold (m_groundFilterDistance);
		seg.setAxis(Eigen::Vector3f(0,0,1));
		seg.setEpsAngle(m_groundFilterAngle);


		PCLPointCloud cloud_filtered(pc);
		// Create the filtering object
		pcl::ExtractIndices<PCLPoint> extract;
		bool groundPlaneFound = false;

		while(cloud_filtered.size() > 10 && !groundPlaneFound){
			seg.setInputCloud(cloud_filtered.makeShared());
			seg.segment (*inliers, *coefficients);
			if (inliers->indices.size () == 0){
				ROS_INFO("PCL segmentation did not find any plane.");

				break;
			}

			extract.setInputCloud(cloud_filtered.makeShared());
			extract.setIndices(inliers);

			if (std::abs(coefficients->values.at(3)) < m_groundFilterPlaneDistance){
				ROS_DEBUG("Ground plane found: %zu/%zu inliers. Coeff: %f %f %f %f", inliers->indices.size(), cloud_filtered.size(),
						coefficients->values.at(0), coefficients->values.at(1), coefficients->values.at(2), coefficients->values.at(3));
				extract.setNegative (false);
				extract.filter (ground);

				// remove ground points from full pointcloud:
				// workaround for PCL bug:
				if(inliers->indices.size() != cloud_filtered.size()){
					extract.setNegative(true);
					PCLPointCloud cloud_out;
					extract.filter(cloud_out);
					nonground += cloud_out;
					cloud_filtered = cloud_out;
				}

				groundPlaneFound = true;
			} else{
				ROS_DEBUG("Horizontal plane (not ground) found: %zu/%zu inliers. Coeff: %f %f %f %f", inliers->indices.size(), cloud_filtered.size(),
						coefficients->values.at(0), coefficients->values.at(1), coefficients->values.at(2), coefficients->values.at(3));
				pcl::PointCloud<PCLPoint> cloud_out;
				extract.setNegative (false);
				extract.filter(cloud_out);
				nonground +=cloud_out;
				// debug
				//            pcl::PCDWriter writer;
				//            writer.write<PCLPoint>("nonground_plane.pcd",cloud_out, false);

				// remove current plane from scan for next iteration:
				// workaround for PCL bug:
				if(inliers->indices.size() != cloud_filtered.size()){
					extract.setNegative(true);
					cloud_out.points.clear();
					extract.filter(cloud_out);
					cloud_filtered = cloud_out;
				} else{
					cloud_filtered.points.clear();
				}
			}

		}
		// TODO: also do this if overall starting pointcloud too small?
		if (!groundPlaneFound){ // no plane found or remaining points too small
			ROS_WARN("No ground plane found in scan");

			// do a rough fitlering on height to prevent spurious obstacles
			pcl::PassThrough<PCLPoint> second_pass;
			second_pass.setFilterFieldName("z");
			second_pass.setFilterLimits(-m_groundFilterPlaneDistance, m_groundFilterPlaneDistance);
			second_pass.setInputCloud(pc.makeShared());
			second_pass.filter(ground);

			second_pass.setFilterLimitsNegative (true);
			second_pass.filter(nonground);
		}

		// debug:
		//        pcl::PCDWriter writer;
		//        if (pc_ground.size() > 0)
		//          writer.write<PCLPoint>("ground.pcd",pc_ground, false);
		//        if (pc_nonground.size() > 0)
		//          writer.write<PCLPoint>("nonground.pcd",pc_nonground, false);

	}


}

void OctomapServer::handlePreNodeTraversal(const ros::Time& rostime){
	if (m_publish2DMap)
	{
		// init projected 2D map:
		m_gridmap.header.frame_id = m_worldFrameId;
		m_gridmap.header.stamp = rostime;
		nav_msgs::MapMetaData oldMapInfo = m_gridmap.info;

		if(m_dynamicmap)
		{
			// TODO: move most of this stuff into c'tor and init map only once (adjust if size changes)
			/*			double minX, minY, minZ, maxX, maxY, maxZ;
						m_octree->getMetricMaxMin(maxX, maxY, maxZ, minX, minY, minZ);
						octomap::point3d minPt(minX, minY, minZ);
						octomap::point3d maxPt(maxX, maxY, maxZ);*/
			/*	octomap::OcTreeKey minKey = m_octree->coordToKey(minPt, m_maxTreeDepth);
				octomap::OcTreeKey maxKey = m_octree->coordToKey(maxPt, m_maxTreeDepth);*/


			// add padding if requested (= new min/maxPts in x&y):
			/*		double halfPaddedX = 0.5*m_minSizeX;
					double halfPaddedY = 0.5*m_minSizeY;
					minX = std::min(minX, -halfPaddedX);
					maxX = std::max(maxX, halfPaddedX);
					minY = std::min(minY, -halfPaddedY);
					maxY = std::max(maxY, halfPaddedY);*/
			/*			minPt = octomap::point3d(minX, minY, minZ);
						maxPt = octomap::point3d(maxX, maxY, maxZ);

						OcTreeKey paddedMaxKey;
						if (!m_octree->coordToKeyChecked(minPt, m_maxTreeDepth, m_paddedMinKey)){
						ROS_ERROR("Could not create padded min OcTree key at %f %f %f", minPt.x(), minPt.y(), minPt.z());
						return;
						}
						if (!m_octree->coordToKeyChecked(maxPt, m_maxTreeDepth, paddedMaxKey)){
						ROS_ERROR("Could not create padded max OcTree key at %f %f %f", maxPt.x(), maxPt.y(), maxPt.z());
						return;
						}

						assert(paddedMaxKey[0] >= maxKey[0] && paddedMaxKey[1] >= maxKey[1]);
			 */
			OcTreeKey paddedMaxKey;
			m_octree->getMinMaxKey(m_paddedMinKey,paddedMaxKey);
			m_multires2DScale = 1 << (m_treeDepth - m_maxTreeDepth);
			int tmp;
			int width = (paddedMaxKey[0] - m_paddedMinKey[0])/m_multires2DScale +2;
			int height = (paddedMaxKey[1] - m_paddedMinKey[1])/m_multires2DScale +2;
			m_gridmap.info.width = width;
			m_gridmap.info.height = height;

			int mapOriginX = 0;
			int mapOriginY = 0;
			assert(mapOriginX >= 0 && mapOriginY >= 0);

			// might not exactly be min / max of octree:
			octomap::point3d origin = m_octree->keyToCoord(m_paddedMinKey, m_treeDepth);
			double gridRes = m_octree->getNodeSize(m_maxTreeDepth);
			m_projectCompleteMap = (!m_incrementalUpdate || (std::abs(gridRes-m_gridmap.info.resolution) > 1e-6));
			m_gridmap.info.resolution = gridRes;
			m_gridmap.info.origin.position.x = origin.x() - gridRes*0.5;
			m_gridmap.info.origin.position.y = origin.y() - gridRes*0.5;
/*			if (m_maxTreeDepth != m_treeDepth){
				m_gridmap.info.origin.position.x -= m_res/2.0;
				m_gridmap.info.origin.position.y -= m_res/2.0;
			}*/

			if(m_completemap)
			{	
				// workaround for  multires. projection not working properly for inner nodes:
				// force re-building complete map
				if (m_maxTreeDepth < m_treeDepth)
					m_projectCompleteMap = true;


				if(m_projectCompleteMap){
					m_gridmap.data.clear();
					// init to unknown:
					m_gridmap.data.resize(m_gridmap.info.width * m_gridmap.info.height, -1);
				} 
				else 
				{

					if (mapChanged(oldMapInfo, m_gridmap.info)){
						adjustMapData(m_gridmap, oldMapInfo);
					}

					nav_msgs::OccupancyGrid::_data_type::iterator startIt;
					size_t mapUpdateBBXMinX = std::max(0, (int(m_updateBBXMin[0]) - int(m_paddedMinKey[0]))/int(m_multires2DScale));
					size_t mapUpdateBBXMinY = std::max(0, (int(m_updateBBXMin[1]) - int(m_paddedMinKey[1]))/int(m_multires2DScale));
					size_t mapUpdateBBXMaxX = std::min(int(m_gridmap.info.width-1), (int(m_updateBBXMax[0]) - int(m_paddedMinKey[0]))/int(m_multires2DScale));
					size_t mapUpdateBBXMaxY = std::min(int(m_gridmap.info.height-1), (int(m_updateBBXMax[1]) - int(m_paddedMinKey[1]))/int(m_multires2DScale));

					assert(mapUpdateBBXMaxX > mapUpdateBBXMinX);
					assert(mapUpdateBBXMaxY > mapUpdateBBXMinY);

					size_t numCols = mapUpdateBBXMaxX-mapUpdateBBXMinX +1;

					// test for max idx:
					uint max_idx = m_gridmap.info.width*mapUpdateBBXMaxY + mapUpdateBBXMaxX;
					if (max_idx  >= m_gridmap.data.size())
						ROS_ERROR("BBX index not valid: %d (max index %zu for size %d x %d) update-BBX is: [%zu %zu]-[%zu %zu]", max_idx, m_gridmap.data.size(), m_gridmap.info.width, m_gridmap.info.height, mapUpdateBBXMinX, mapUpdateBBXMinY, mapUpdateBBXMaxX, mapUpdateBBXMaxY);

					// reset proj. 2D map in bounding box:
					for (unsigned int j = mapUpdateBBXMinY; j <= mapUpdateBBXMaxY; ++j){
						std::fill_n(m_gridmap.data.begin() + m_gridmap.info.width*j+mapUpdateBBXMinX,
								numCols, -1);
					}

				}
			}
			else
			{
				adjustMapData(m_gridmap, oldMapInfo);
			}
		}
		else
		{
		}
	}

}

void OctomapServer::handlePostNodeTraversal(const ros::Time& rostime){
	if (m_publish2DMap)
	{
//		octomap_msgs::Pos getpos;
//		setOrigin(getpos);
//		setend(getpos);	
//		getpos.yaw=yaw;
//		int originx=(pow(2,15)-m_paddedMinKey[0])/m_multires2DScale;
//		int originy=(pow(2,15)-m_paddedMinKey[1])/m_multires2DScale;
//		getpos.originx=originx;
//		getpos.originy=originy;
//		getpos.startx-=originx;
//		getpos.starty-=originy;
//		getpos.endx-=originx;
//		getpos.endy-=originy;
//		ros::Time now=ros::Time::now();
//		getpos.header.stamp=now;
		m_gridmap.header.stamp=rostime;
		m_gridmap.header.frame_id=m_worldFrameId;	
//		pospub.publish(getpos);	
		m_mapPub.publish(m_gridmap);
//		delOrigin();
//		delend();	
	}
}

void OctomapServer::handleOccupiedNode(const OcTreeT::iterator& it){

	if (m_publish2DMap && m_projectCompleteMap){
		update2DMap(it, true);
	}
}

void OctomapServer::handleFreeNode(const OcTreeT::iterator& it){

	if (m_publish2DMap && m_projectCompleteMap){
		update2DMap(it, false);
	}
}


void OctomapServer::handleOccupiedNodeInBBX(const OcTreeT::iterator& it){

	if (m_publish2DMap && !m_projectCompleteMap){
		update2DMap(it, true);
	}
}

void OctomapServer::handleFreeNodeInBBX(const OcTreeT::iterator& it){

	if (m_publish2DMap && !m_projectCompleteMap){
		update2DMap(it, false);
	}
}

void OctomapServer::update2DMap(const OcTreeT::iterator& it, bool occupied){

	// update 2D map (occupied always overrides):
	if (it.getDepth() == m_maxTreeDepth){
		unsigned idx = mapIdx(it.getKey());
		if (occupied)
			m_gridmap.data[mapIdx(it.getKey())] = 100;
		else if (m_gridmap.data[idx] == -1){
			m_gridmap.data[idx] = 0;
		}

	} else{
		int intSize = 1 << (m_maxTreeDepth - it.getDepth());
		octomap::OcTreeKey minKey=it.getIndexKey();
		for(int dx=0; dx < intSize; dx++){
			int i = (minKey[0]+dx - m_paddedMinKey[0])/m_multires2DScale;
			for(int dy=0; dy < intSize; dy++){
				unsigned idx = mapIdx(i, (minKey[1]+dy - m_paddedMinKey[1])/m_multires2DScale);
				if (occupied)
					m_gridmap.data[idx] = 100;
				else if (m_gridmap.data[idx] == -1){
					m_gridmap.data[idx] = 0;
				}
			}
		}
	}


}

void OctomapServer::update2DMap(const octomap::upKey::iterator& it, bool occupied)
{
	int i=(it->key[0]-m_paddedMinKey[0])/m_multires2DScale;
	int j=(it->key[1]-m_paddedMinKey[1])/m_multires2DScale;
	int idx = mapIdx(i,j);
	if(idx>0&&idx<m_gridmap.data.size())
	{
		if (occupied)
		{
			m_gridmap.data[idx] = 100;
			int starti=i-box;
			int stopi=i+box;
			if(starti<0)
				starti=0;
			if(stopi>=m_gridmap.info.width)
				stopi=m_gridmap.info.width-1;
			int diff=stopi-starti;
			int startj=j-box;
			int stopj=j+box;
			if(startj<0)
				startj=0;
			if(stopj>=m_gridmap.info.height)
				stopj=m_gridmap.info.height-1;
			for(;startj<=stopj;startj++)
			{
				int ss=mapIdx(starti,startj);
				for(int tmpi=0;tmpi<=diff;tmpi++)
				{
					if(m_gridmap.data[ss]!=100)
						m_gridmap.data[ss]=40;		
					ss++;
				}
			}
		}
		else if (m_gridmap.data[idx] == -1){
			m_gridmap.data[idx] = 0;
		}
	}
}

void OctomapServer::clearupdatemap(int maxx,int maxy,int minx,int miny)
{
	unsigned mapminx=(minx-m_paddedMinKey[0]) / m_multires2DScale;
	unsigned mapmaxx=(maxx-m_paddedMinKey[0]) / m_multires2DScale;
	unsigned width=mapmaxx-mapminx;
	unsigned mapminy=(miny-m_paddedMinKey[1]) / m_multires2DScale;
	unsigned mapmaxy=(maxy-m_paddedMinKey[1]) / m_multires2DScale;
	unsigned height=mapmaxy-mapminy;
	for(int i=0;i<height;i++)
	{
		unsigned tmpx=mapminx;
		unsigned tmpy=mapminy+i;
		unsigned mapstart=mapIdx(tmpx,tmpy);
		for(int j=0;j<width;j++)
		{
			int tmpid=mapstart+j;
			if(m_gridmap.data[tmpid]!=60)
				m_gridmap.data[tmpid]=-1;
		}
	}
}

void OctomapServer::setOrigin(octomap_msgs::Pos& pos)
{
	octomap::OcTreeKey point;
	m_octree->coordToKeyChecked(sensorOrigin, point);	
	int i=(point[0]-m_paddedMinKey[0])/m_multires2DScale;
	int j=(point[1]-m_paddedMinKey[1])/m_multires2DScale;
	pos.startx=i;
	pos.starty=j;
	int idx=mapIdx(i,j);
	m_gridmap.data[idx]=20;

}

void OctomapServer::delOrigin()
{
	octomap::OcTreeKey point;
	m_octree->coordToKeyChecked(sensorOrigin, point);	
	int idx=mapIdx(point);
	if(flag==0)
		m_gridmap.data[idx]=0;
	else
		m_gridmap.data[idx]=60;

}

void OctomapServer::setend(octomap_msgs::Pos& pos)
{
	octomap::OcTreeKey point;
	octomap::point3d tmp(anend[0],anend[1],anend[2]);
	m_octree->coordToKeyChecked(tmp,point);
	int i=(point[0]-m_paddedMinKey[0])/m_multires2DScale;
	int j=(point[1]-m_paddedMinKey[1])/m_multires2DScale;
	pos.endx=i;
	pos.endy=j;
	int idx=mapIdx(i,j);
	if(idx>0&&idx<m_gridmap.data.size())
		m_gridmap.data[idx]=80;
}
void OctomapServer::delend()
{
	octomap::OcTreeKey point;
	octomap::point3d tmp(anend[0],anend[1],anend[2]);
	m_octree->coordToKeyChecked(tmp,point);
	int idx=mapIdx(point);
	m_gridmap.data[idx]=0;
}


void OctomapServer::setbyhuman(const std_msgs::String::ConstPtr& msg)
{

	string a=msg->data.c_str();
	vector<string> words;
	string word;
	int index = a.find(',');
	float x[2];
	int j;
	a[index]=' ';
	stringstream ss(a);
	for(j=0;j<2;j++)
	{
		ss>>x[j];
	}
	cout<<x[0]<<endl;
	cout<<"before "<<sensorOrigin.x()<<" "<<sensorOrigin.y()<<endl;
	float x1=sensorOrigin.x()+x[0];
	float y1=sensorOrigin.y()+x[1];
	anend[0]=x1;anend[1]=y1;anend[2]=sensorOrigin.z();

	flag=1;
}

void OctomapServer::savemap(const std_msgs::String::ConstPtr& msg)
{
	string tmp=msg->data.c_str();
	int robot,cmd;
	string filename;
	stringstream ss(tmp);
	ss>>robot;
	ss>>cmd;
	if(robot==robot_id && cmd==4)
	{
		octomap_msgs::Octomap octreemsg;	
		octreemsg.header.frame_id = m_worldFrameId;
		octreemsg.header.stamp = ros::Time::now();
		octreemsg.robot=robot_id;
		if (!octomap_msgs::binaryMapToMsg(*m_octree, octreemsg))
			return;
		octreepub.publish(octreemsg);
	}
}



bool OctomapServer::isSpeckleNode(const OcTreeKey&nKey) const {
	OcTreeKey key;
	bool neighborFound = false;
	for (key[2] = nKey[2] - 1; !neighborFound && key[2] <= nKey[2] + 1; ++key[2]){
		for (key[1] = nKey[1] - 1; !neighborFound && key[1] <= nKey[1] + 1; ++key[1]){
			for (key[0] = nKey[0] - 1; !neighborFound && key[0] <= nKey[0] + 1; ++key[0]){
				if (key != nKey){
					OcTreeNode* node = m_octree->search(key);
					if (node && m_octree->isNodeOccupied(node)){
						// we have a neighbor => break!
						neighborFound = true;
					}
				}
			}
		}
	}

	return neighborFound;
}

void OctomapServer::reconfigureCallback(octomap_server::OctomapServerConfig& config, uint32_t level){
	if (m_maxTreeDepth != unsigned(config.max_depth))
		m_maxTreeDepth = unsigned(config.max_depth);
	else{
		m_pointcloudMinZ            = config.pointcloud_min_z;
		m_pointcloudMaxZ            = config.pointcloud_max_z;
		m_occupancyMinZ             = config.occupancy_min_z;
		m_occupancyMaxZ             = config.occupancy_max_z;
		m_filterSpeckles            = config.filter_speckles;
		m_filterGroundPlane         = config.filter_ground;
		m_groundFilterDistance      = config.ground_filter_distance;
		m_groundFilterAngle         = config.ground_filter_angle;
		m_groundFilterPlaneDistance = config.ground_filter_plane_distance;
		m_maxRange                  = config.sensor_model_max_range;
		m_compressMap               = config.compress_map;
		m_incrementalUpdate         = config.incremental_2D_projection;

		m_octree->setProbHit(config.sensor_model_hit);
		m_octree->setProbMiss(config.sensor_model_miss);
		m_octree->setClampingThresMin(config.sensor_model_min);
		m_octree->setClampingThresMax(config.sensor_model_max);
	}
	publishAll();
}

void OctomapServer::adjustMapData(nav_msgs::OccupancyGrid& map, const nav_msgs::MapMetaData& oldMapInfo) const{
	if (map.info.resolution != oldMapInfo.resolution){
		ROS_ERROR("Resolution of map changed, cannot be adjusted");
		return;
	}


	int i_off = int((oldMapInfo.origin.position.x - map.info.origin.position.x)/map.info.resolution +0.5);
	int j_off = int((oldMapInfo.origin.position.y - map.info.origin.position.y)/map.info.resolution +0.5);


	if (i_off < 0 || j_off < 0
			|| oldMapInfo.width  + i_off > map.info.width
			|| oldMapInfo.height + j_off > map.info.height)
	{
		ROS_ERROR("New 2D map does not contain old map area, this case is not implemented");
		return;
	}

	nav_msgs::OccupancyGrid::_data_type oldMapData = map.data;


	map.data.clear();
	// init to unknown:
	map.data.resize(map.info.width * map.info.height, -1);

	nav_msgs::OccupancyGrid::_data_type::iterator fromStart, fromEnd, toStart;

	for (int j =0; j < int(oldMapInfo.height); ++j ){
		// copy chunks, row by row:
		fromStart = oldMapData.begin() + j*oldMapInfo.width;
		fromEnd = fromStart + oldMapInfo.width;
		toStart = map.data.begin() + ((j+j_off)*m_gridmap.info.width + i_off);
		copy(fromStart, fromEnd, toStart);
	}

}


std_msgs::ColorRGBA OctomapServer::heightMapColor(double h) {

	std_msgs::ColorRGBA color;
	color.a = 1.0;
	// blend over HSV-values (more colors)

	double s = 1.0;
	double v = 1.0;

	h -= floor(h);
	h *= 6;
	int i;
	double m, n, f;

	i = floor(h);
	f = h - i;
	if (!(i & 1))
		f = 1 - f; // if i is even
	m = v * (1 - s);
	n = v * (1 - s * f);

	switch (i) {
		case 6:
		case 0:
			color.r = v; color.g = n; color.b = m;
			break;
		case 1:
			color.r = n; color.g = v; color.b = m;
			break;
		case 2:
			color.r = m; color.g = v; color.b = n;
			break;
		case 3:
			color.r = m; color.g = n; color.b = v;
			break;
		case 4:
			color.r = n; color.g = m; color.b = v;
			break;
		case 5:
			color.r = v; color.g = m; color.b = n;
			break;
		default:
			color.r = 1; color.g = 0.5; color.b = 0.5;
			break;
	}

	return color;
}
}



