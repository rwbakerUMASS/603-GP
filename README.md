# CS-603 Group Project: Human Following Robot
Kevin Antony Gomez, Ryan Baker, and Chunru Lin


To Set Up:
1. package.xml contains all dependencies, runnning rosdep will install all required packages
2. Requires making a change to lfilter.launch and filters.yaml inside map_laser package
    1. remove remap line from base_scan to scan
    2. change filter name and type to:\
    name: shadows
    type: laser_filters/ScanShadowsFilter
3. catkin_make
4. source devel/setup.bash
5. gmapping
    1. roslaunch stingray_camera triton.launch
    2. roslaunch stingray_camera triton_gmapping.launch
    3. Drive around
    4. rosrun map_server map_saver -f map
    5. Copy map to 603-GP/map/

To Run:
1. On Trition:
    1. cd stingray_docker; sudo docker-compose up
    2. cd stingray_docker; sudo docker-compose exec triton_noetic bash
    3. source devel/setup.bash
    4. roslaunch stingray_camera triton.launch
2. On remote computer:
    1. roslaunch 603-GP follower.launch
