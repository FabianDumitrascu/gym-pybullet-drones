from acados_template import AcadosModel
from casadi import SX, MX, vertcat, horzcat, sin, cos, diag
import numpy as np

def exportModel():
  mass = 0.027
  model_name = "quadrotor_test_model"
  nx = 13
  ixx = 1.4e-5
  iyy = 1.4e-5
  izz = 2.17e-5
  kf = 3.16e-10
  km = 7.94e-12


  # set up states
  x1 = MX.sym('x1')   # xpos
  x2 = MX.sym('x2')   # ypos
  x3 = MX.sym('x3')   # zpos
  x4 = MX.sym('x4')   # xVel
  x5 = MX.sym('x5')   # yVel
  x6 = MX.sym('x6')   # zVel
  x7 = MX.sym('x7')   # Qw
  x8 = MX.sym('x8')   # Qx
  x9 = MX.sym('x9')   # Qy
  x10 = MX.sym('x10') # Qz
  x11 = MX.sym('x11')  # omegax
  x12 = MX.sym('x12')  # omegay
  x13 = MX.sym('x13')  # omegaz
  x = vertcat(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13)

  xdot = MX.sym('xdot', nx, 1)

  # set up controls
  u1 = MX.sym('u1')  # motor thrust_0
  u2 = MX.sym('u2')  # motor thrust_1
  u3 = MX.sym('u3')  # motor thrust_2
  u4 = MX.sym('u4')  # motor thrust_3
  u = vertcat(u1, u2, u3, u4)

  km_kf = km / kf

  moment_x = 0.0397 * (u1 - u2 - u3 + u4)
  moment_y = 0.0397 * (-u1 - u2 + u3 + u4)

  f_expl_rpm = vertcat(x4, x5, x6, \
                       (1/mass)*(u1+u2+u3+u4)*(2*x7*x9 + 2*x8*x10)/(x7**2 + x8**2 + x9**2 + x10**2),
                       -(1/mass)*(u1+u2+u3+u4)*(2*x7*x8 - 2*x9*x10)/(x7**2 + x8**2 + x9**2 + x10**2),
                       -(1/mass)*(u1+u2+u3+u4)*(2*x8*x8 + 2*x9*x9 - 1)/(x7**2 + x8**2 + x9**2 + x10**2) - 9.81,
                       -(x11*x8)/2 - (x12*x9)/2 - (x13*x10)/2,
                       (x11*x7)/2 + (x13*x9)/2 - (x12*x10)/2,
                       (x12*x7)/2 - (x13*x8)/2 + (x11*x10)/2,
                       (x13*x7)/2 + (x12*x8)/2 - (x11*x9)/2,
                       (moment_x + iyy*x12*x13 - izz*x12*x13)/ixx,
                       (moment_y - ixx*x11*x13 + izz*x11*x13)/iyy,
                       (km_kf*(u1-u2+u3-u4) + ixx*x11*x12 - iyy*x11*x12)/izz)

  f_expl = f_expl_rpm

  # parameters
  ext_param = MX.sym('references', nx + 13, 1) # ext_param = MX.sym('references', nx + 4, 1)
  p = []
  z = []

  model = AcadosModel()
  model.f_impl_expr = xdot - f_expl
  model.f_expl_expr = f_expl
  model.x = x
  model.u = u
  model.xdot = xdot
  model.z = z
  model.p = p
  model.name = "quadrotor"

  return model