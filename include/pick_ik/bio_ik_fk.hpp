#pragma once

#include <pick_ik_plugin_export.h>
#include <functional>
#include <vector>
#include <string>
#include <mutex>
#include <Eigen/Geometry>
#include <memory>
#include <moveit/robot_model/robot_model.h>

namespace pick_ik {

using FkFn = std::function<std::vector<Eigen::Isometry3d>(std::vector<double> const&)>;

PICK_IK_PLUGIN_EXPORT
auto make_bio_ik_fk_fn(std::shared_ptr<moveit::core::RobotModel const> robot_model,
                     const moveit::core::JointModelGroup* jmg,
                     const std::vector<std::string>& tip_link_names,
                     std::mutex& mx) -> FkFn;
}

#include <Eigen/Dense>
#include <memory>
#include <moveit/robot_model/robot_model.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_kdl/tf2_kdl.hpp>

#include <emmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>

#if(__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9))
#define FUNCTION_MULTIVERSIONING 1
#else
#define FUNCTION_MULTIVERSIONING 0
#endif

namespace pick_ik::bio_ik_fk {

typedef tf2::Quaternion Quaternion;
typedef tf2::Vector3 Vector3;

struct alignas(32) Frame
{
    Vector3 pos;
    double __padding[4 - (sizeof(Vector3) / sizeof(double))];
    Quaternion rot;
    inline Frame() {}
    inline Frame(const tf2::Vector3& p, const tf2::Quaternion& r)
        : pos(p)
        , rot(r)
    {
    }
    explicit inline Frame(const KDL::Frame& kdl)
    {
        pos = tf2::Vector3(kdl.p.x(), kdl.p.y(), kdl.p.z());
        double qx, qy, qz, qw;
        kdl.M.GetQuaternion(qx, qy, qz, qw);
        rot = tf2::Quaternion(qx, qy, qz, qw);
    }
    explicit inline Frame(const geometry_msgs::msg::Pose& msg)
    {
        tf2::fromMsg(msg.orientation, rot);
        pos = tf2::Vector3(msg.position.x, msg.position.y, msg.position.z);
    }
    explicit inline Frame(const Eigen::Isometry3d& f)
    {
        pos = tf2::Vector3(f.translation().x(), f.translation().y(), f.translation().z());
        Eigen::Quaterniond q(f.rotation());
        rot = tf2::Quaternion(q.x(), q.y(), q.z(), q.w());
    }

    inline const Vector3& getPosition() const { return pos; }
    inline const Quaternion& getOrientation() const { return rot; }
    inline void setPosition(const Vector3& p) { pos = p; }
    inline void setOrientation(const Quaternion& q) { rot = q; }

    inline Eigen::Isometry3d toIsometry3d() const
    {
        return Eigen::Translation3d(pos.x(), pos.y(), pos.z()) *
               Eigen::Quaterniond(rot.w(), rot.x(), rot.y(), rot.z());
    }

private:
    template <size_t i> struct IdentityFrameTemplate
    {
        static const Frame identity_frame;
    };

public:
    static inline const Frame& identity() { return IdentityFrameTemplate<0>::identity_frame; }
};

template <size_t i>
const Frame Frame::IdentityFrameTemplate<i>::identity_frame(Vector3(0, 0, 0), Quaternion(0, 0, 0, 1));

__attribute__((always_inline)) inline void quat_mul_vec(const tf2::Quaternion& q, const tf2::Vector3& v, tf2::Vector3& r)
{
    double v_x = v.x();
    double v_y = v.y();
    double v_z = v.z();

    double q_x = q.x();
    double q_y = q.y();
    double q_z = q.z();
    double q_w = q.w();

    if((v_x == 0 && v_y == 0 && v_z == 0) || (q_x == 0 && q_y == 0 && q_z == 0 && q_w == 1))
    {
        r = v;
        return;
    }

    double t_x = q_y * v_z - q_z * v_y;
    double t_y = q_z * v_x - q_x * v_z;
    double t_z = q_x * v_y - q_y * v_x;

    double r_x = q_w * t_x + q_y * t_z - q_z * t_y;
    double r_y = q_w * t_y + q_z * t_x - q_x * t_z;
    double r_z = q_w * t_z + q_x * t_y - q_y * t_x;

    r_x += r_x;
    r_y += r_y;
    r_z += r_z;

    r_x += v_x;
    r_y += v_y;
    r_z += v_z;

    r.setX(r_x);
    r.setY(r_y);
    r.setZ(r_z);
}

__attribute__((always_inline)) inline void quat_mul_quat(const tf2::Quaternion& p, const tf2::Quaternion& q, tf2::Quaternion& r)
{
    double p_x = p.x();
    double p_y = p.y();
    double p_z = p.z();
    double p_w = p.w();

    double q_x = q.x();
    double q_y = q.y();
    double q_z = q.z();
    double q_w = q.w();

    double r_x = (p_w * q_x + p_x * q_w) + (p_y * q_z - p_z * q_y);
    double r_y = (p_w * q_y - p_x * q_z) + (p_y * q_w + p_z * q_x);
    double r_z = (p_w * q_z + p_x * q_y) - (p_y * q_x - p_z * q_w);
    double r_w = (p_w * q_w - p_x * q_x) - (p_y * q_y + p_z * q_z);

    r.setX(r_x);
    r.setY(r_y);
    r.setZ(r_z);
    r.setW(r_w);
}

__attribute__((always_inline)) inline void concat(const Frame& a, const Frame& b, Frame& r)
{
    tf2::Vector3 d;
    quat_mul_vec(a.rot, b.pos, d);
    r.pos = a.pos + d;
    quat_mul_quat(a.rot, b.rot, r.rot);
}

__attribute__((always_inline)) inline void concat(const Frame& a, const Frame& b, const Frame& c, Frame& r)
{
    Frame tmp;
    concat(a, b, tmp);
    concat(tmp, c, r);
}

__attribute__((always_inline)) inline void quat_inv(const tf2::Quaternion& q, tf2::Quaternion& r)
{
    r.setX(-q.x());
    r.setY(-q.y());
    r.setZ(-q.z());
    r.setW(q.w());
}

__attribute__((always_inline)) inline void invert(const Frame& a, Frame& r)
{
    Frame tmp;
    quat_inv(a.rot, r.rot);
    quat_mul_vec(r.rot, -a.pos, r.pos);
}

__attribute__((always_inline)) inline void change(const Frame& a, const Frame& b, const Frame& c, Frame& r)
{
    Frame tmp;
    invert(b, tmp);
    concat(a, tmp, c, r);
}

// computes and caches local joint frames
class RobotJointEvaluator
{
private:
    std::vector<double> joint_cache_variables;
    std::vector<Frame> joint_cache_frames;
    std::vector<Frame> link_frames;

protected:
    std::vector<Vector3> joint_axis_list;
    moveit::core::RobotModelConstPtr robot_model;
    std::vector<double> variables;

public:
    inline void getJointFrame(const moveit::core::JointModel* joint_model, const double* vars, Frame& frame)
    {
        auto joint_type = joint_model->getType();
        if(joint_type == moveit::core::JointModel::FIXED)
        {
            frame = Frame::identity();
            return;
        }
        size_t joint_index = joint_model->getJointIndex();
        switch(joint_type)
        {
        case moveit::core::JointModel::REVOLUTE:
        {
            auto axis = joint_axis_list[joint_index];

            auto v = vars[joint_model->getFirstVariableIndex()];
            auto half_angle = v * 0.5;
            auto fcos = cos(half_angle);
            auto fsin = sin(half_angle);

            frame = Frame(Vector3(0.0, 0.0, 0.0), Quaternion(axis.x() * fsin, axis.y() * fsin, axis.z() * fsin, fcos));

            break;
        }
        case moveit::core::JointModel::PRISMATIC:
        {
            auto axis = joint_axis_list[joint_index];
            auto v = vars[joint_model->getFirstVariableIndex()];
            frame = Frame(axis * v, Quaternion(0.0, 0.0, 0.0, 1.0));
            break;
        }
        case moveit::core::JointModel::FLOATING:
        {
            auto* vv = vars + joint_model->getFirstVariableIndex();
            frame.pos = Vector3(vv[0], vv[1], vv[2]);
            frame.rot = Quaternion(vv[3], vv[4], vv[5], vv[6]).normalized();
            break;
        }
        default:
        {
            auto* joint_variables = vars + joint_model->getFirstVariableIndex();
            Eigen::Isometry3d joint_transform;
            joint_model->computeTransform(joint_variables, joint_transform);
            frame = Frame(joint_transform);
            break;
        }
        }
    }
    inline void getJointFrame(const moveit::core::JointModel* joint_model, const std::vector<double>& vars, Frame& frame) { getJointFrame(joint_model, vars.data(), frame); }

protected:
    inline const Frame& getLinkFrame(const moveit::core::LinkModel* link_model) { return link_frames[link_model->getLinkIndex()]; }

    inline bool checkJointMoved(const moveit::core::JointModel* joint_model)
    {
        size_t i0 = joint_model->getFirstVariableIndex();
        size_t cnt = joint_model->getVariableCount();
        if(cnt == 0) return true;
        if(cnt == 1) return !(variables[i0] == joint_cache_variables[i0]);
        for(size_t i = i0; i < i0 + cnt; i++)
            if(!(variables[i] == joint_cache_variables[i])) return true;
        return false;
    }

    inline const Frame& getJointFrame(const moveit::core::JointModel* joint_model)
    {
        size_t joint_index = joint_model->getJointIndex();

        if(!checkJointMoved(joint_model)) return joint_cache_frames[joint_index];

        getJointFrame(joint_model, variables, joint_cache_frames[joint_index]);

        size_t cnt = joint_model->getVariableCount();
        if(cnt)
        {
            size_t i0 = joint_model->getFirstVariableIndex();
            if(cnt == 1)
                joint_cache_variables[i0] = variables[i0];
            else
                for(size_t i = i0; i < i0 + cnt; i++)
                    joint_cache_variables[i] = variables[i];
        }

        return joint_cache_frames[joint_index];
    }

public:
    RobotJointEvaluator(moveit::core::RobotModelConstPtr model)
        : robot_model(model)
    {
        joint_cache_variables.clear();
        joint_cache_variables.resize(model->getVariableCount(), DBL_MAX);

        joint_cache_frames.clear();
        joint_cache_frames.resize(model->getJointModelCount());

        link_frames.resize(model->getLinkModelCount());
        for(auto* link_model : model->getLinkModels())
            link_frames[link_model->getLinkIndex()] = Frame(link_model->getJointOriginTransform());

        joint_axis_list.clear();
        joint_axis_list.resize(robot_model->getJointModelCount());
        for(size_t i = 0; i < joint_axis_list.size(); i++)
        {
            auto* joint_model = robot_model->getJointModel(i);
            if(auto* j = dynamic_cast<const moveit::core::RevoluteJointModel*>(joint_model)) joint_axis_list[i] = Vector3(j->getAxis().x(), j->getAxis().y(), j->getAxis().z());
            if(auto* j = dynamic_cast<const moveit::core::PrismaticJointModel*>(joint_model)) joint_axis_list[i] = Vector3(j->getAxis().x(), j->getAxis().y(), j->getAxis().z());
        }
    }
};

// fast tree fk
class RobotFK_Fast_Base : protected RobotJointEvaluator
{
protected:
    std::vector<std::string> tip_names;
    std::vector<Frame> tip_frames;
    std::vector<const moveit::core::LinkModel*> tip_links;
    std::vector<const moveit::core::LinkModel*> link_schedule;
    std::vector<Frame> global_frames;
    std::vector<std::vector<const moveit::core::LinkModel*>> link_chains;

    inline void updateMimic(std::vector<double>& values)
    {
        for(auto* joint : robot_model->getMimicJointModels())
        {
            auto src = joint->getMimic()->getFirstVariableIndex();
            auto dest = joint->getFirstVariableIndex();
            values[dest] = values[src] * joint->getMimicFactor() + joint->getMimicOffset();
        }
    }

public:
    RobotFK_Fast_Base(moveit::core::RobotModelConstPtr model)
        : RobotJointEvaluator(model)
    {
    }

    void initialize(const std::vector<std::string>& tip_link_names)
    {
        tip_names = tip_link_names;
        tip_frames.resize(tip_names.size());

        tip_links.clear();
        for(const auto& n : tip_names)
            tip_links.push_back(robot_model->getLinkModel(n));

        global_frames.resize(robot_model->getLinkModelCount());

        link_chains.clear();
        link_schedule.clear();
        for(auto* tip_link : tip_links)
        {
            std::vector<const moveit::core::LinkModel*> chain;
            for(auto* link = tip_link; link; link = link->getParentLinkModel())
                chain.push_back(link);
            reverse(chain.begin(), chain.end());
            link_chains.push_back(chain);
            for(auto* link : chain)
            {
                if(find(link_schedule.begin(), link_schedule.end(), link) != link_schedule.end()) continue;
                link_schedule.push_back(link);
            }
        }
    }

    void applyConfiguration(const std::vector<double>& jj0)
    {
        variables = jj0;
        updateMimic(variables);
        for(auto* link_model : link_schedule)
        {
            auto* joint_model = link_model->getParentJointModel();
            auto* parent_link_model = joint_model->getParentLinkModel();
            if(parent_link_model)
            {
                concat(global_frames[parent_link_model->getLinkIndex()], getLinkFrame(link_model), getJointFrame(joint_model), global_frames[link_model->getLinkIndex()]);
            }
            else
            {
                concat(getLinkFrame(link_model), getJointFrame(joint_model), global_frames[link_model->getLinkIndex()]);
            }
        }
        for(size_t itip = 0; itip < tip_links.size(); itip++)
        {
            tip_frames[itip] = global_frames[tip_links[itip]->getLinkIndex()];
        }
    }

    inline const std::vector<Frame>& getTipFrames() const { return tip_frames; }
};

}  // namespace pick_ik::bio_ik_fk 