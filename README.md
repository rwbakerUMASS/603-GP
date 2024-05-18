Requires people -> leg_detector package to function.

Requires making a change to lfilter.launch and filters.yaml inside map_laser package
remove remap line from base_scan to scan, change filter name and type to: 
    name: shadows
    type: laser_filters/ScanShadowsFilter