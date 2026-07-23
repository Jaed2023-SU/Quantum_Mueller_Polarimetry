import numpy as np
from scipy.optimize import minimize  # type: ignore

np.set_printoptions(precision = 3, suppress = True)
epsilon = 10e-3

def Rotation_matrix(Theta_degree):
    Theta = np.deg2rad(Theta_degree)
    Rotation = np.eye(2)
    Rotation = [[np.cos(2 * Theta), np.sin(2*Theta)],
                [-np.sin(2 * Theta), np.cos(2 * Theta)]]
    return Rotation

def QWP_rot(fast_axis_angle_degree):
    
    Rotation = np.eye(4)
    Rotation[1:3, 1:3] = Rotation_matrix(fast_axis_angle_degree)
    
    QWP = np.eye(4)
    QWP[2:4, 2:4] = [[np.cos(np.pi / 2), np.sin(np.pi / 2)],
                     [-np.sin(np.pi / 2), np.cos(np.pi / 2)]]
    return Rotation.T @ QWP @ Rotation

def HWP_rot(fast_axis_angle_degree):
    
    Rotation = np.eye(4)
    Rotation[1:3, 1:3] = Rotation_matrix(2 * fast_axis_angle_degree)
    
    HWP = np.eye(4)
    HWP[2:4, 2:4] = [[np.cos(np.pi), np.sin(np.pi)],
                     [-np.sin(np.pi), np.cos(np.pi)]]
    return Rotation.T @ HWP @ Rotation
    
def comp_angle(Samples, Target = np.eye(4), initial_guess = (10.0, 10.0, 10.0)):
    
    Sample = np.asarray(Samples, dtype = float)
    Identity = Target

    bounds = [(0, 135), (0, 67.5), (0, 135)]
    # bounds = [(-45, 45), (-22.5, 45), (-45, 45)]

    def objective(angles):
        QWP_angle_1, HWP_angle, QWP_angle_2 = angles
        # QWP_angle_1, HWP_angle = angles

        Calculated = QWP_rot(QWP_angle_2) @ HWP_rot(HWP_angle) @ QWP_rot(QWP_angle_1) @ Sample
        # Calculated = HWP_rot(HWP_angle) @ QWP_rot(QWP_angle_1) @ Sample
        residual = Calculated - Identity
        return np.linalg.norm(residual, ord="fro") ** 2

    result = minimize(
        objective,
        x0 = np.asarray(initial_guess, dtype = float),
        method = "L-BFGS-B",
        bounds = bounds)

    QWP_angle_1, HWP_angle, QWP_angle_2 = result.x
    # QWP_angle_1, HWP_angle = result.x

    calculated_matrix = (QWP_rot(QWP_angle_2) @ HWP_rot(HWP_angle) @ QWP_rot(QWP_angle_1))
    #calculated_matrix = (HWP_rot(HWP_angle) @ QWP_rot(QWP_angle_1))

    residual_norm = np.linalg.norm(calculated_matrix @ Sample - Identity, ord = "fro") ** 2

    return result, calculated_matrix, residual_norm

###########################################################
SS =  QWP_rot(72.22) @ HWP_rot(-12.0) @ QWP_rot(14.0) @ HWP_rot(32.0)
SS_comp_meas = np.linalg.inv(SS)
Target = QWP_rot(45)
# Target = np.eye(4)
print("Sample matrix")
print(SS)
initial_guess = np.asarray([0.0, 0.0, 0.0])
Total_result, SS_comp, res_comp = comp_angle(SS, Target = Target, initial_guess = initial_guess)
iteration = 0
while res_comp > epsilon:
    iteration += 1
    initial_guess += np.asarray([10, 10, 10])
    # initial_guess += np.asarray(10 * (np.random.rand(3)))
    Total_result, SS_comp, res_comp = comp_angle(SS, Target = Target, initial_guess = initial_guess)
    print("Iteration", iteration)
    if iteration > 13:
        break
print("Calculated matrix: Must be same with Corrected")
print(SS_comp)
print("Target matrix")
print(Target)
print("Compensated: Must be Target")
print(SS_comp @ SS)
print(res_comp)
print(Total_result.x)
print("Iteration", iteration)