#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <lib/json.h>

#include "vehicle.hpp"
#include "parse.hpp"

namespace cuauv {
namespace conf {

drag_plane_vector parse_drag_planes(const Json::Value& arr)
{
    if (!arr.isArray()) {
        throw std::runtime_error("Expected array of drag planes.");
    }

    drag_plane_vector dps;
    for (unsigned int i = 0; i < arr.size(); i++) {
        Json::Value root = arr[i];
        if (!root.isObject()) {
            throw std::runtime_error("Expected drag plane object.");
        }
        PARSE_VECTOR3(pos);
        PARSE_VECTOR3(normal);
        PARSE_DOUBLE(cD);
        PARSE_DOUBLE(area);
        drag_plane dp { pos, normal, cD, area };
        dps.push_back(dp);
    }

    return dps;
}

thruster_vector parse_thrusters(const Json::Value& arr)
{
    if (!arr.isArray()) {
        throw std::runtime_error("Expected array of thrusters.");
    }

    thruster_vector ts;
    for (unsigned int i = 0; i < arr.size(); i++) {
        Json::Value root = arr[i];
        if (!root.isObject()) {
            throw std::runtime_error("Expected thruster object.");
        }
        PARSE_STRING(name);
        PARSE_VECTOR3(pos);
        PARSE_TWO_DOUBLES(heading_pitch, yaw, pitch);
        thruster t { name, pos, yaw, pitch };
        ts.push_back(t);
    }

    return ts;
}

camera_vector parse_cameras(const Json::Value& arr)
{
    if (!arr.isObject()) {
        throw std::runtime_error("Expected list of cameras.");
    }

    camera_vector cs;
    for (Json::ValueIterator itr = arr.begin(); itr != arr.end(); itr++) {
        Json::Value root = *itr;
        if (!root.isObject()) {
            throw std::runtime_error("Expected camera object.");
        }
        std::string tag = itr.key().asString();
        PARSE_STRING(type);
        PARSE_INTEGER(id);
        PARSE_STRING(camera_name);
        PARSE_INTEGER(width);
        PARSE_INTEGER(height);
        PARSE_VECTOR3(position);
        PARSE_VECTOR3(orientation_hpr);
        PARSE_TWO_DOUBLES(sensor_size_wh_mm, sensor_width, sensor_height);
        PARSE_DOUBLE(focal_length_mm);
        camera c { tag, type, id, camera_name, width, height, position,
                   orientation_hpr, sensor_width, sensor_height, focal_length_mm };
        cs.push_back(c);
    }

    return cs;
}

vehicle load_vehicle(void)
{
    char* dir = getenv("CUAUV_SOFTWARE");
    if (dir == nullptr) {
        throw std::runtime_error("cuauv::conf::load_vehicle: CUAUV_SOFTWARE must be set to the root of the software repository, with a trailing slash.");
    }

    char* vehicle_env_maybe = getenv("CUAUV_VEHICLE");
    std::string vehicle_env = vehicle_env_maybe == nullptr ? "thor" : std::string(vehicle_env_maybe);

    std::string dirs(dir);
    if (dirs[dirs.size() - 1] != '/') {
        dirs = dirs + "/";
    }
    dirs = dirs + "conf/";

    // std::string filename(std::string(vehicle_env) + ".conf");
    std::string filename(vehicle_env + ".conf");

    std::ifstream ifs(dirs + filename);
    if (!ifs) {
        throw std::runtime_error("cuauv::conf::load_vehicle: Failed to open vehicle file.");
    }

    Json::Value root;
    Json::Reader reader;
    if (!reader.parse(ifs, root)) {
        throw std::runtime_error("cuauv::conf::load_vehicle: Failed to parse vehicle file.");
    }

    PARSE_VECTOR3(center_of_buoyancy)
    PARSE_DOUBLE(buoyancy_force)
    PARSE_DOUBLE(gravity_force)
    PARSE_DOUBLE(sub_height)
    PARSE_MATRIX(I)
    PARSE_VECTOR3(Ib)
    PARSE_QUATERNION(btom_rq) btom_rq.normalize();
    PARSE_DRAG_PLANES(drag_planes)
    PARSE_VECTOR6(cwhe_axes)
    PARSE_VECTOR6(thruster_minimums)
    PARSE_VECTOR6(thruster_maximums)
    PARSE_DRAG_PLANES(uncompensated_drag_planes)
    PARSE_THRUSTERS(thrusters)
    PARSE_CAMERAS(cameras)

    vehicle v {
        center_of_buoyancy,
        buoyancy_force,
        gravity_force,
        sub_height,
        I,
        Ib,
        btom_rq,
        drag_planes,
        cwhe_axes,
        thruster_minimums,
        thruster_maximums,
        uncompensated_drag_planes,
        thrusters,
        cameras
    };
    
    return v;
}

} // namespace conf
} // namespace cuauv
