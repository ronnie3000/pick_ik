#include <pick_ik/bio_ik_fk.hpp>

#include <vector>
#include <string>
#include <mutex> // Added for std::mutex

namespace pick_ik {

auto make_bio_ik_fk_fn(std::shared_ptr<moveit::core::RobotModel const> robot_model,
                     const moveit::core::JointModelGroup* jmg,
                     const std::vector<std::string>& tip_link_names,
                     std::mutex& mx) -> std::function<std::vector<Eigen::Isometry3d>(std::vector<double> const&)> {

    auto fk_solver = std::make_shared<bio_ik_fk::RobotFK_Fast_Base>(robot_model);
    fk_solver->initialize(tip_link_names);
    
    // Get the all joint variables for the model
    auto all_joint_variables = robot_model->getVariableNames();

    return [=, &mx](std::vector<double> const& active_positions) mutable {
        std::scoped_lock lock(mx);
        // BioIK expects a vector of all joint positions, not just the active ones.
        // We need to map the active joint positions to the full joint vector.
        std::vector<double> all_positions;
        all_positions.resize(all_joint_variables.size(), 0.0);

        auto const& active_joint_variables = jmg->getVariableNames();
        for (size_t i = 0; i < active_joint_variables.size(); ++i) {
            auto it = std::find(all_joint_variables.begin(), all_joint_variables.end(), active_joint_variables[i]);
            if (it != all_joint_variables.end()) {
                all_positions[static_cast<size_t>(std::distance(all_joint_variables.begin(), it))] = active_positions[i];
            }
        }
        
        fk_solver->applyConfiguration(all_positions);
        
        const auto& tip_frames_bio_ik = fk_solver->getTipFrames();
        
        std::vector<Eigen::Isometry3d> tip_frames;
        tip_frames.reserve(tip_frames_bio_ik.size());
        for (const auto& frame : tip_frames_bio_ik) {
            tip_frames.push_back(frame.toIsometry3d());
        }
        
        return tip_frames;
    };
}

}  // namespace pick_ik 