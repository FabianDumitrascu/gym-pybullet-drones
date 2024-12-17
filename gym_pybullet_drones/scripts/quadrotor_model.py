from acados_template import AcadosModel
from casadi import SX, vertcat, horzcat, sin, cos, diag
import numpy as np

def export_quadrotor_model() -> AcadosModel:

    model_name = 'quadrotor'

    # constants #################################################################
    g = vertcat(0,0, -9.81)
    m = 0.027
    const = np.sqrt(2)/2
    l = 0.0397
    cq = 7.94e-12
    ct = 3.16e-10
    G = np.array([[1, 1, 1, 1],
    			  [-l * const, -l * const, l * const, l * const],
    		      [-l * const, l * const, l * const, -l * const],
    		      [-cq/ct, cq/ct, -cq/ct, cq/ct]])
    zb = np.array([0,0,-1]).T
    Ixx = 1.4e-5
    Iyy = 1.4e-5
    Izz = 2.17e-5

    # set up states & controls #################################################
    x       = SX.sym('x')
    y   	= SX.sym('y')
    z      = SX.sym('z')
    x_dot = SX.sym('x_dot')
    y_dot 	= SX.sym('y_dot')
    z_dot     = SX.sym('z_dot')
    qw       = SX.sym('qw')
    qx   	= SX.sym('qx')
    qy      = SX.sym('qy')
    qz		= SX.sym('qz')
    omega_x	= SX.sym('omega_x')
    omega_y = SX.sym('omega_y')
    omega_z = SX.sym('omega_z')
    
    x = vertcat(x, y, z, x_dot, y_dot, z_dot, qw, qx, qy, qz, omega_x, omega_y, omega_z)

    print(f"State vector (x) dimension: {x.shape}")
    

    rot_s_1 = SX.sym('rot_s_1')
    rot_s_2 = SX.sym('rot_s_2')
    rot_s_3 = SX.sym('rot_s_3')
    rot_s_4 = SX.sym('rot_s_4')
	
    u = vertcat(rot_s_1, rot_s_2, rot_s_3, rot_s_4)

    print(f"Control vector (u) dimension: {u.shape}")
	
    # xdot ##################################################################################
    x_dot       = SX.sym('x_dot')
    y_dot   	= SX.sym('y_dot')
    z_dot      = SX.sym('z_dot')
    x_ddot = SX.sym('x_ddot')
    y_ddot 	= SX.sym('y_ddot')
    z_ddot     = SX.sym('z_ddot')
    qw_dot       = SX.sym('qw_dot')
    qx_dot   	= SX.sym('qx_dot')
    qy_dot      = SX.sym('qy_dot')
    qz_dot		= SX.sym('qz_dot')
    omega_x_dot	= SX.sym('omega_x_dot')
    omega_y_dot = SX.sym('omega_y_dot')
    omega_z_dot = SX.sym('omega_z_dot')
    
    xdot = vertcat(x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot, qw_dot, qx_dot, qy_dot, qz_dot, omega_x_dot, omega_y_dot, omega_z_dot)

    # Dynamics ################################################################################################
    R = vertcat(
        horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)),
        horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2))
    )
	
    T1 = ct * rot_s_1**2
    T2 = ct * rot_s_2**2
    T3 = ct * rot_s_3**2
    T4 = ct * rot_s_4**2

    tau1 = cq * rot_s_1**2
    tau2 = cq * rot_s_2**2
    tau3 = cq * rot_s_3**2
    tau4 = cq * rot_s_4**2
   
    thrust_total = vertcat(T1 + T2 + T3 + T4)

    thrust_in_inertial = thrust_total * R @ vertcat(0, 0, 1)
    
    omega_B = vertcat(omega_x, omega_y, omega_z)

    tau_x = -l * const * (T1 + T2) + l * const * (T3 + T4)
    tau_y = -l * const * (T1 + T4) + l * const * (T2 + T3)
    tau_z = -tau1 + tau2 - tau3 + tau4

    torque = vertcat(tau_x, tau_y, tau_z)


    q_dot = 0.5 * vertcat(
        -qx * omega_x - qy * omega_y - qz * omega_z,
        qw * omega_x - qz * omega_y + qy * omega_z,
        qz * omega_x + qw * omega_y - qx * omega_z,
        -qy * omega_x + qx * omega_y + qw * omega_z
    )

    q_norm = SX.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw = qw / q_norm
    qx = qx / q_norm
    qy = qy / q_norm
    qz = qz / q_norm

    Iv = diag(vertcat(Ixx, Iyy, Izz))
    Iv_inv = diag(vertcat(1/Ixx, 1/Iyy, 1/Izz))
    
    gyro_term = vertcat(
        omega_y * (Iv @ omega_B)[2] - omega_z * (Iv @ omega_B)[1],
        omega_z * (Iv @ omega_B)[0] - omega_x * (Iv @ omega_B)[2],
        omega_x * (Iv @ omega_B)[1] - omega_y * (Iv @ omega_B)[0]
    )

    omega_dot = Iv_inv @ (torque - gyro_term)

    x_acc = (thrust_in_inertial[0]) / m
    y_acc = (thrust_in_inertial[1]) / m 
    z_acc = (thrust_in_inertial[2]) / m + g[2]

    f_expl = vertcat(x_dot,
                     y_dot,
                     z_dot,
                     x_acc,
                     y_acc,
                     z_acc,
                    q_dot[0],       
                    q_dot[1],      
                    q_dot[2],       
                    q_dot[3],       
                    omega_dot[0],   
                    omega_dot[1],   
                    omega_dot[2]    
    )
    

    f_impl = xdot - f_expl

    print(f"Explicit dynamics (f_expl) dimension: {f_expl.shape}")
    print(f"Implicit dynamics (f_impl) dimension: {f_impl.shape}")

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # store meta information
    #model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    #model.u_labels = ['$F$']
    #model.t_label = '$t$ [s]'

    return model