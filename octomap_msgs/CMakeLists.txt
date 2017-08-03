cmake_minimum_required(VERSION 2.8)
project(octomap_msgs)

find_package(catkin REQUIRED COMPONENTS roscpp rospy message_generation geometry_msgs std_msgs sensor_msgs)

include_directories(include)

add_message_files(
  DIRECTORY msg
  FILES Octomap.msg OctomapWithPose.msg info.msg Pos.msg
  )

add_service_files(
  DIRECTORY srv
  FILES GetOctomap.srv BoundingBoxQuery.srv
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs sensor_msgs)

catkin_package(DEPENDS message_runtime geometry_msgs std_msgs sensor_msgs
               INCLUDE_DIRS include
)

install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
        PATTERN ".svn" EXCLUDE)
