import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import sys
import time

# Set up the colors
spdred = (177/255, 4/255, 14/255)
spdblue = (0/255, 124/255, 146/255)
spdblack = (0, 0, 0)

# Switches to turn on/off a few applications
write = 1 # Write the percentage of the progress on running the code
gif = 1 # Creates a gif (takes some time depending on the parameters; has to be used with movie)
energy = 0 # Plot the average energy per particle per iteration
velocity = 0 # Plot the velocity distribution before and after evolving the gas
collision = 0 # If collision = 1 the particles will collide with each other; collision = 0 and the particles will collide with the walls
uniform = 0 # If uniform = 0, the particles velocities will be initiated as sampled from a gaussian
show_init = 0 # If 1, plot the particles initial positions in the case with collisions

fig = plt.figure()

niter = 2000 # Number of time steps (dt in line 148)

npart = 200 # Number of particles; there is a close relation between the number of particles with their sizes
L0 = 0.0 # Left/bottom limit of the square box
L1 = 1.0 # Right/top limit of the square box
#L1 = 6.342e-11 # Right/top limit of the square box in the realistic case
pi = 3.14159265359 # Just to set the area of the particles to fill the box
psize = 0.01 # "Particle radius" for the gif
#psize = np.sqrt((0.05*(L1-L0)*(L1-L0))/(npart*pi)) # The 0.05 is to ensure that the particles occupy 5% of the volume of the box
col0 = L0+psize # Left/bottom limit for collision
col1 = L1-psize # Right/top limit for collision
colp = 2.0*psize # Collision parameter for particles
kB = 1.38065e-23 # Boltzmann constant in J/K
R = 8.314 # Universal gas constant in J/(K.mol)
vel = 1.0 # Particles velocity parameter (maximum/minimum of uniform, and necessary for gaussian standard deviation)
#vel = 533.0 # N2 velocity in m/s (maximum/minimum of uniform, and necessary for gaussian standard deviation)
m = 1.0 # Particles mass
#m = 4.6534e-26 # N2 molecule mass in kg
V = (L1-L0)*(L1-L0) # Box volume

points_radius = psize/0.0022 # "Right" particle radius (can be buggy)

# Defines the colors of the particles in the gif (around 20% of the particles have a different color than the rest)
color = []
color_array = np.zeros(npart)
rand = np.random.randint(0,npart,int(npart/5))
randc = []
randc.append(rand)
for c in randc:
	color_array[c] = 1
for c in range(npart):
	if(color_array[c]==1):
		color.append(spdred)
	else:
		color.append(spdblue)

npart_inside = 0 # Will be used to calculate the number of particles inside the box after the simulation (unfortunately, some might vanish)

# Defines the particles x and y positions for the collisionless case (the +/- 1.0/1000.0 is to be sure that the particles are initiated inside the walls)
if(collision==0):
	pposx = np.random.uniform((L0+(1.0/1000.0)),(L1-(1.0/1000.0)),npart)
	pposy = np.random.uniform((L0+(1.0/1000.0)),(L1-(1.0/1000.0)),npart)

# This block defines the particles x and y positions for the case with collisions; particles are initiated at the same distance, and a random term is added to their positions
elif(collision==1):

	sqnpart = np.round(np.sqrt(npart))
	pposx = np.zeros(npart)
	pposy = np.zeros(npart)

	pdist = (col1-col0)/(np.sqrt(npart))
	varpdist = (pdist-colp)/4.0

	if(pdist<=colp):
		print("Your particles are too big. Try reducing the amount of particles or their size")
		sys.exit()

	nmax = sqnpart
	nc=0
	for i in range(npart):
		pposx[i] = col0 + (nc)*pdist
		if((i+1)%(nmax)==0):
			nc+=1

	inc=0
	for i in range(npart):
		pposy[i] = col0 + (i-inc)*pdist
		if((i+1)%(nmax)==0):
			inc=i+1

	pposx = pposx + (1.0-(np.max(pposx)+psize))/2.0
	pposy = pposy + (1.0-(np.max(pposy)+psize))/2.0
	checkposx0 = pposx>=(col0+varpdist)
	checkposx1 = pposx<=(col1-varpdist)
	pposx = pposx + (checkposx0*checkposx1)*np.random.uniform(-varpdist,varpdist,npart)
	checkposy0 = pposy>=(col0+varpdist)
	checkposy1 = pposy<=(col1-varpdist)
	pposy = pposy + (checkposx0*checkposx1)*np.random.uniform(-varpdist,varpdist,npart)

	# This block calculates the distance between each pair of particles (to see if the particles are still overlapping) ############################################

	pposxaux = np.repeat(pposx,npart)
	pposxaux = pposxaux.reshape(npart,npart)
	pposyaux = np.repeat(pposy,npart)
	pposyaux = pposyaux.reshape(npart,npart)

	dx = pposxaux - pposx
	dy = pposyaux - pposy
	d = np.sqrt((dx*dx) + (dy*dy))
	np.fill_diagonal(d,2)

	if(np.sum(d<=colp)>0):
		print("Your particles are too big. Try reducing the amount of particles or their size")
		sys.exit()

	print('The percentage of the box volume occupied by particles is {:.1f}%'.format((npart*psize*psize*pi*100.0)/((L1-L0)*(L1-L0))))

#################################################################################

	if(show_init==1):
		plt.scatter(pposx,pposy,s=points_radius**2)
		plt.subplots_adjust(bottom=0.1, right=0.72, top=0.9)
		plt.ylim(L0,L1)
		plt.xlim(L0,L1)
		plt.show()

	if((L1-max(pposx))<psize or (L1-max(pposy))<psize or (min(pposx)-L0)<psize or (min(pposy)-L0)<psize):
		print("Your particles are too big. Try reducing the amount of particles or their size")
		sys.exit()

# Defines the particles x and y velocities (if uniform or gaussian)
if(uniform==1):
	pvelx = np.random.uniform(-vel,vel,npart)
	pvely = np.random.uniform(-vel,vel,npart)
	ptype = "uniform"
elif(uniform==0):
	pvelx = np.random.normal(0.0,0.56*vel,npart) # This mean/standard deviation is to be consistent with the uniform velocities in the collision case
	pvely = np.random.normal(0.0,0.56*vel,npart) # This mean/standard deviation is to be consistent with the uniform velocities in the collision case
	ptype = "gaussian"

# Variable to name the images
if(collision==0):
	ifcol="nocol"
elif(collision==1):
	ifcol="col"

# Creates lists to store the particles velocity information to create the velocity distributions
y_i = [] # Module of initial velocity
y_f = [] # Module of final velocity
y_ix = [] # Initial velocity in x axis
y_fx = [] # Final velocity in x axis
y_iy = [] # Initial velocity in y axis
y_fy = [] # Final velocity in y axis

vmax = np.max(np.sqrt((pvelx*pvelx)+(pvely*pvely))) # Calculates the maximum module of the particles velocities
print("The maximum initial velocity is (approximately): {:.4f} m/s".format(vmax))

vmean = np.mean(((pvelx*pvelx)+(pvely*pvely))) # Calculates the maximum module of the particles velocities
print("The mean initial velocity is (approximately): {:.4f} m/s".format(vmean))

# The following if is to store the particles starting velocities
if(velocity==1):
	y_i.append(np.sqrt((pvelx*pvelx)+(pvely*pvely))) # Initial module of velocity
	y_ix.append(pvelx) # Initial x velocity 
	y_iy.append(pvely) # Initial y velocity

dt = psize/(5.0*vmax) # Time interval (the 500 is just to set a smaller interval to prevent particles from leaving the box)
print("The time step is (approximately): {:.4f} s".format(dt))

# Set up formatting for the movie files (setting the gifs to have 5 seconds)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=int(niter/10.0), metadata=dict(artist='Me'), bitrate=1800)

ims = [] # Image list for the gif

# Lists to calculate energy when activated
x = [] 
y_e = []

pvel = np.zeros(npart) # Will be used to calculate the module of the particles velocities

Ei = np.mean((pvelx*pvelx) + (pvely*pvely))*m*npart/2.0 # Initial average energy

# This block collides particles with the walls and with each other and updates their positions/velocities ###################################################

for i in range(niter):

	pvel += np.sqrt((pvelx**2 + pvely**2)) # Calculates the module of the particles velocities

	x.append(i)

	# The following bolck will only be used if the particles collide with each other
	if(collision == 1):

		# Auxiliary arrays to calculate the particles distance
		pposxaux = np.repeat(pposx,npart)
		pposxaux = pposxaux.reshape(npart,npart)
		pposyaux = np.repeat(pposy,npart)
		pposyaux = pposyaux.reshape(npart,npart)

		# Calculates the distance between each pair of particles
		dx = pposxaux - pposx
		dy = pposyaux - pposy
		d = np.sqrt((dx*dx) + (dy*dy))
		np.fill_diagonal(d,2)

		# Auxiliary objects to separate the particles
		min0 = np.min(d,axis=0)
		argmin0 = np.argmin(d,axis=0)
		colpp = np.zeros(npart)

		listaux = [] # Auxiliary list to separate the particles position

		# This "for" finds the closest pair to each particle and store it in a list
		for p in range(npart):
			minaux = np.min(min0*(argmin0==p) + (argmin0!=p))
			argminaux = np.argmin(min0 + (argmin0!=p))
			listaux.append([minaux,argminaux])
			colpp[p] = minaux

		listaux.sort() # If there are two particles that are the closest to a third one, this will sort by the pair with the smallest distance

		# This "if" is just to see if the total number of particles is even or odd (this makes a small difference in what follows)
		if(npart%2==0):
			npartaux = npart
		else:
			npartaux = npart-1

		# This "for" is to, if the distance between a pair is smaller than colp, update the position and velocity of the particles in the pair
		for j in range(0,npartaux,2):

			if(listaux[j][0]==listaux[j+1][0] and listaux[j][0]<colp):

				x1 = np.copy(pposx[listaux[j][1]])
				x2 = np.copy(pposx[listaux[j+1][1]])
				y1 = np.copy(pposy[listaux[j][1]])
				y2 = np.copy(pposy[listaux[j+1][1]])
				vx1 = np.copy(pvelx[listaux[j][1]])
				vx2 = np.copy(pvelx[listaux[j+1][1]])
				vy1 = np.copy(pvely[listaux[j][1]])
				vy2 = np.copy(pvely[listaux[j+1][1]])

				# Updating the velocity
				pvelx[listaux[j][1]] = vx1 - ((((vx1-vx2)*(x1-x2))+((vy1-vy2)*(y1-y2)))/(((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2))))*(x1-x2)
				pvely[listaux[j][1]] = vy1 - ((((vx1-vx2)*(x1-x2))+((vy1-vy2)*(y1-y2)))/(((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2))))*(y1-y2)
				pvelx[listaux[j+1][1]] = vx2 - ((((vx2-vx1)*(x2-x1))+((vy2-vy1)*(y2-y1)))/(((x2-x1)*(x2-x1))+((y2-y1)*(y2-y1))))*(x2-x1)
				pvely[listaux[j+1][1]] = vy2 - ((((vx2-vx1)*(x2-x1))+((vy2-vy1)*(y2-y1)))/(((x2-x1)*(x2-x1))+((y2-y1)*(y2-y1))))*(y2-y1)

				# Updating the position (there will be a second update in the position after checking if the particle will collide with the wall; this is to prevent a bug where a pair of particles would stick together and behave as only one in a very strange way)
				if(pposx[listaux[j][1]] < pposx[listaux[j+1][1]]):
					pposx[listaux[j][1]] = pposx[listaux[j][1]] - (colp - listaux[j][0])
				elif(pposx[listaux[j][1]] > pposx[listaux[j+1][1]]):
					pposx[listaux[j][1]] = pposx[listaux[j][1]] + (colp - listaux[j][0])

	# Elastically collides the particles with the wall if the particle x or y position is smaller than col0 or col1, respectively, by updating its velocity; the pressure per particle is being calculated and added up as well

	if(collision==0):

		colpx0 = (pposx+dt*pvelx)<=L0
		pvelx = (colpx0*(-2))*pvelx + pvelx
		pposx = pposx + (colpx0*(L0 - pposx))
		colpx1 = (pposx+dt*pvelx)>=L1
		pvelx = (colpx1*(-2))*pvelx + pvelx
		pposx = pposx - (colpx1*(pposx - L1))
		colpy0 = (pposy+dt*pvely)<=L0
		pvely = (colpy0*(-2))*pvely + pvely
		pposy = pposy + (colpy0*(L0 - pposy))
		colpy1 = (pposy+dt*pvely)>=L1
		pvely = (colpy1*(-2))*pvely + pvely
		pposy = pposy - (colpy1*(pposy - L1))

	if(collision==1):

		colpx0 = (pposx+dt*pvelx)<=col0
		pvelx = (colpx0*(-2))*pvelx + pvelx
		pposx = pposx + (colpx0*(col0 - pposx))
		colpx1 = (pposx+dt*pvelx)>=col1
		pvelx = (colpx1*(-2))*pvelx + pvelx
		pposx = pposx - (colpx1*(pposx - col1))
		colpy0 = (pposy+dt*pvely)<=col0
		pvely = (colpy0*(-2))*pvely + pvely
		pposy = pposy + (colpy0*(col0 - pposy))
		colpy1 = (pposy+dt*pvely)>=col1
		pvely = (colpy1*(-2))*pvely + pvely
		pposy = pposy - (colpy1*(pposy - col1))

	# Update the particles positions with respect to the ammount they travel in a time interval dt
	pposx = pposx + pvelx*dt
	pposy = pposy + pvely*dt

	# Calculates the number of particles that are still inside the box (some vanish and I am trying to correct this bug)
	npart_inside = (pposx>L0)*(pposx<L1)*(pposy>L0)*(pposy<L1)

	# Calculates the energy per particle per iteration and store it in the list (considering that the particles have mass = 1 kg)
	if(energy==1):
		y_e.append((np.mean((pvelx*pvelx) + (pvely*pvely)))*m/(2.0)) # E/N

	# store the images in the list to create the gif
	if(gif==1):
		im = plt.scatter(pposx,pposy,s=points_radius**2,c=color)
		plt.subplots_adjust(bottom=0.1, right=0.72, top=0.9)
		plt.ylim(L0,L1)
		plt.xlim(L0,L1)
		plt.Rectangle((L0, L0), L1, L1, fill=False, edgecolor='black', linewidth=2.5)
		ims.append([im])

	# Writes how many iterations passed (in percentage)
	if(write==1):
		if(niter>=100):
			if((i+1)%(niter/100) == 0):    
				print('{}%/100% -- Simulation time elapsed: {:.4f} s'.format(int((i/(niter/100))+1),((i+1)*dt)))

print("The number of particles inside the box is {} (sometimes a few particles escape; trying to fix that though)".format(np.sum(npart_inside))) # Some particles escape

######################################################################################################################################## 

Ef = np.mean((pvelx*pvelx) + (pvely*pvely))*m*npart/2.0 # Final average energy

print('Initial and final internal energy are {} and {}, respectively.'.format(Ei,Ef))

P = (np.mean(pvelx*pvelx)+np.mean(pvely*pvely))*m*(npart/(2.0*V)) # Pressure of the gas calculated using the kinetic theory of gases
Teq = (np.mean((pvelx*pvelx)+(pvely*pvely)))*m/(2.0*kB) # Temperature calculated through the equipartition theorem
Tid = (P*V)/((npart/6.02e23)*8.314) # Temperature calculated through the ideal gases law

print("The pressure is: {:.4f}".format(P))
print("The temperature calculated using the equipartition theorem is: {}".format(Teq))
print("The temperature calculated using the ideal gases law is: {}".format(Tid))
print("The difference in the temperatures is: {:.4f}%".format(abs((Teq-Tid)/Tid)*100.0))

# Saves the gif
if(gif==1):
	ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,repeat_delay=100)
	ani.save(str(npart)+'part_'+str(niter)+'iter_'+ptype+'_particle_box_'+ifcol+'.mp4', writer=writer)
	plt.clf()

# Writes the histograms of the particles velocities
if(velocity==1):
	y_f.append(np.sqrt((pvelx*pvelx)+(pvely*pvely)))
	y_fx.append(pvelx)
	y_fy.append(pvely)

	pveli, bins, _ = plt.hist(y_i, bins=100, range=[-0.01, 4.0*vel], histtype = 'step', density=False, label='Particles initial velocity', color = spdred)
	pvelf = plt.hist(y_f, bins=bins, histtype = 'step', density=False, label='Particles final velocity', color = spdblue)
	plt.xlabel('Velocity')
	v = (bins[1:] + bins[:-1])/2
    # The mean kinetic energy per Particle.
	KE = np.mean((np.mean((pvelx*pvelx)+(pvely*pvely))) / 2)
    # The Maxwell-Boltzmann equilibrium distribution of speeds.
	a = 1.0/2.0/KE
	f = 2*a*v*np.exp(-a*v**2)
	f = np.sum(pvelf[0])*(f/np.sum(f))
	plt.plot(v, f,label='Theoretical Maxwell-Boltzmann distribution', color = spdblack)
	plt.legend(loc=1)
	plt.savefig(str(npart)+'part_'+str(niter)+'iter_'+ptype+'_'+ifcol+'_particles_velocity.pdf', format='pdf')
	plt.clf()

	_, bins, _ = plt.hist(y_ix, bins=100, range=[-4.0*vel, 4.0*vel], histtype = 'step', density=False, label='Particles initial x velocity', color = spdred)
	gaussyn = plt.hist(y_fx, bins=bins, histtype = 'step', density=False, label='Particles final x velocity', color = spdblue)
	plt.xlabel('Velocity')
	plt.legend(loc=1)
	plt.savefig(str(npart)+'part_'+str(niter)+'iter_'+ptype+'_'+ifcol+'_particles_x_velocity.pdf', format='pdf')
	plt.clf()

	_, bins, _ = plt.hist(y_iy, bins=100, range=[-4.0*vel, 4.0*vel], histtype = 'step', density=False, label='Particles initial y velocity', color = spdred)
	gaussyn = plt.hist(y_fy, bins=bins, histtype = 'step', density=False, label='Particles final y velocity', color = spdblue)
	plt.xlabel('Velocity')
	plt.legend(loc=1)
	plt.savefig(str(npart)+'part_'+str(niter)+'iter_'+ptype+'_'+ifcol+'_particles_y_velocity.pdf', format='pdf')
	plt.clf()

# Writes the graph of the energy per particle per iteration
if(energy==1):
	print("The min and max energy are {} and {} respectively.".format(min(y_e),max(y_e)))
	plt.plot(x,y_e,label='Average energy per particle',color=spdred)
	plt.ylim(min(y_e)-0.1*min(y_e),max(y_e)+0.1*min(y_e))
	plt.xlabel('Iteration')
	plt.ylabel('Energy')
	plt.legend(loc=1)
	plt.savefig(str(npart)+'part_'+str(niter)+'iter_'+ptype+'_'+ifcol+'_particles_energy.pdf', format='pdf')
	plt.clf()
