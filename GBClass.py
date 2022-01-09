from numpy import pi
import numpy as np
from PIL import Image
import cv2

class MonochromaticLight:
	def __init__(self, w_0, λ, Lx, Ly, Nx=100):
		self.z_R=pi*w_0**2/λ

		self.w= lambda z: w_0*np.sqrt( 1+(z/self.z_R)**2 )	# beam width
		self.R= lambda z: z + self.z_R**2/z 	# real radius of curvature
		self.P= lambda z: 1j + np.log( w_0/self.w(z) ) - 1j*np.arctan( z/self.z_R )	# gouy phase shift
		self.q= lambda z: 1/( 1/self.R(z) + 1j*λ/(pi*self.w(z)**2) )	# complex Radius of Curvature

		self.k=2*pi/λ	# wave number

		x=np.linspace(-Lx/2,Lx/2,Nx)
		y=np.linspace(-Ly/2,Ly/2,int(Nx*Ly/Lx))
		x, y = np.meshgrid(x,y)
		self.r2=x**2+y**2

		self.u=lambda z: np.exp( 1j*( self.P(z) + self.k*self.r2/(2*self.q(z)) ) )*np.exp(1j*self.k*z)
		self.I=lambda z: np.abs(self.u(z))**2
		self.I_scaled=lambda z: 255*self.I(z)/np.max(self.I(z))




λ=700e-9
frames=[]
res=100
L=0.5
image=np.zeros((res,res,3))
light=MonochromaticLight(1e-7, λ, *[.3,.3], res)

for i in np.linspace(50,40,1000,-1):
	GBRed=np.abs(light.u(L+i*λ)+light.u(L))**2
	GBRed=255*GBRed/np.max(GBRed)
	frames.append(GBRed.astype(np.uint8))


out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'mp4v'), 60, (res,res),False)

for i in range(len(frames)):
    out.write(frames[i])
out.release()







