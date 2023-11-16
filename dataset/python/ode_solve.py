import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import consts as c

BACKSPIN = 1
TOPSPIN = -1

r = 3.3E-2;  # ball radius
A = np.pi * r ** 2;  # ball csa
d = 1.21;  # air density (kg/m^3)
m = 58E-3;  # mass in kg
# cd = 0.55;  # for a new ball (change later to formula based on CL)
# cl = 0;  # change later to formula
g = 9.807;  # gravitational acceleration


def calc_cl(v, vspin, spin_dir):
	if vspin > 0:
		return spin_dir * (1 / (2 + v / vspin))
	else:
		return 0


def calc_cd(v, vspin):
	if vspin > 0:
		return 0.55 + 1 / (22.5 + 4.2 * (v / vspin) ** (2.5)) ** (0.4)
	else:
		return 0.55


def solve_numeric(v0, elev, azimuth, spin, spin_dir, x0, y0, z0, start, end, num_points, wind_v, wind_azimuth):
	elev_r = elev * (np.pi / 180);  # launch elevation in radians
	azimuth_r = azimuth * (np.pi / 180);  # launch azimuth in radians
	wind_azimuth_r = wind_azimuth * (np.pi / 180);  # wind azimuth in radians

	cd_w = 0.55  # drag coefficient for wind
	vx_w = wind_v * np.sin(wind_azimuth_r)
	vy_w = wind_v * np.cos(wind_azimuth_r)

	vx0 = v0 * np.cos(elev_r) * np.sin(azimuth_r)  # initial velocity
	vy0 = v0 * np.cos(elev_r) * np.cos(azimuth_r)  # initial y velocity
	vz0 = v0 * np.sin(elev_r)
	vspin = r * spin * 2 * np.pi / 60

	# initial condition
	z_init = [vx0 * m, x0, vy0 * m, y0, vz0 * m, z0]

	# time points
	t = np.linspace(start, end, num=num_points)

	def model(x, t):
		vx = x[0] / m - vx_w
		vy = x[2] / m - vy_w
		vz = x[4] / m

		v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
		v_p = np.sqrt(vx ** 2 + vy ** 2)

		cd = calc_cd(v, vspin)  # coefficient of drag
		cl = calc_cl(v, vspin, spin_dir)  # coefficient of lift

		dPxdt = (-A * d * v / 2) * (cl * vz * np.sin(azimuth_r) + cd * vx)
		dxdt = vx + vx_w
		dPydt = (-A * d * v / 2) * (cl * vz * np.cos(azimuth_r) + cd * vy)
		dydt = vy + vy_w
		dPzdt = (A * d * v / 2) * (cl * v_p - cd * vz) - m * g
		dzdt = vz

		# v_arr.append(dzdt)
		return [dPxdt, dxdt, dPydt, dydt, dPzdt, dzdt]

	# solve ODE
	z = odeint(model, z_init, t)
	return z	


if __name__ == "__main__":
	v0=23
	elev=8
	azimuth=-10
	spin=0
	spin_dir=TOPSPIN
	x0=0
	y0=0
	z0=1
	start=0
	end=5
	num_points=1000
	wind_v=0
	wind_azimuth=90

	z1 = solve_numeric(v0, elev, azimuth, spin, spin_dir, x0, y0, z0, start, end, num_points, wind_v, wind_azimuth)
	# z2 = solve_numeric(25, 8.1, 0, 0, TOPSPIN, 0, 0, 1, 0, 5, 1000, 10, 90)
	# z3 = solve_numeric(25, 8.1, 6, 0, BACKSPIN, 0, 0, 1, 0, 5, 1000, 10, 90)

	z = [z1]

	fig = plt.figure('points_3d',figsize=(15*1.5,4*1.5))
	ax = fig.add_subplot(111, projection='3d')

	# plot results
	for z_arr in z:
		z_arr = z_arr[z_arr[:, 5] > 0]
		ax.plot3D(z_arr[:, 1]*c.SCALER, z_arr[:, 3]*c.SCALER, z_arr[:, 5]*c.SCALER)
	   
	ax.plot3D(c.TENNIS_X_POINTS,c.TENNIS_Y_POINTS,c.TENNIS_Z_POINTS, 'black')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(c.XMIN, c.XMAX)
	ax.set_ylim(0.01, c.YMAX)
	ax.set_zlim(0.01, c.ZMAX)

	ax.view_init(elev=25,azim=-15)

	if spin == 0:
		spin_dir_str=""
	else:
		spin_dir_str = "\nBackspin" if spin_dir is BACKSPIN else "\nFrontspin"
	text_str = f"Velocity = {v0} m/s\nElevation = {elev}°\nAzimuth = {azimuth}°\nSpin = {spin} rpm{spin_dir_str}"

	# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	# ax.text(5.2, 7.5, 1.4, s=text_str, transform=ax.transAxes, fontsize=14, bbox=props)

	# ax.legend(['Azimuth = -6°', 'Azimuth = 0°', 'Azimuth = 6°'])
	plt.show()
