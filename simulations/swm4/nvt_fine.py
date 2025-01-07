# This script was generated by OpenMM-Setup on 2021-07-22.

import sys
try:
    from simtk.openmm import *
    from simtk.openmm.app import *
    from simtk.unit import *
except ModuleNotFoundError:
    from openmm import *
    from openmm.app import *
    from openmm.unit import *
import parmed
from velocityverletplugin import VVIntegrator
sys.path.append('/home/andras/velrep')
from velreporter import VELFile, VELReporter
#from openmmtools.integrators import VelocityVerletIntegrator

cnt = int(sys.argv[1])
pcnt = cnt-1

if cnt == 1:
    rst = f'traj/nvt_20.rst'

if cnt > 1:
    rst = f'traj/nvt_fine_{pcnt}.rst'

# Input Files

psf = CharmmPsfFile('swm4_4000.psf')
crd = CharmmCrdFile('swm4_4000_charmm.cor')


paramsfile = 'ff.str'
parFiles = ()
for line in open(paramsfile, 'r'):
    if '!' in line: line = line.split('!')[0]
    parfile = line.strip()
    if len(parfile) != 0: parFiles += ( parfile, )

params = CharmmParameterSet( *parFiles )

# System Configuration

nonbondedMethod = PME
nonbondedCutoff = 1.2*nanometers
ewaldErrorTolerance = 0.0005
constraints = HBonds # SHAKE
rigidWater = True
constraintTolerance = 0.000001

# Integration Options

dt = 0.0002*picoseconds
temperature = 300*kelvin
friction = 1.0/picosecond
pressure = 1.0*atmospheres
barostatInterval = 25
coll_freq = 10.0/picosecond
drud_coll_freq = 200.0/picosecond
drud_temp = 1*kelvin

# Simulation Options

platform = Platform.getPlatformByName('CUDA')
platformProperties = {'Precision': 'mixed'}

minim = float('inf')
maxim = float('-inf')
for i in range(len(crd.positions)):
    for j in range(3):
        if crd.positions[i]._value[j] > maxim:
            maxim = crd.positions[i]._value[j]
        if crd.positions[i]._value[j] < minim:
            minim = crd.positions[i]._value[j]

xtl = (maxim - minim)*angstroms
psf.setBox(xtl,xtl,xtl)
topology = psf.topology
positions = crd.positions

#ff = ForceField('parmed_omm.xml')
#system = ff.createSystem(psf.topology,
system = psf.createSystem(params,
    nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
    constraints=constraints, rigidWater=rigidWater, ewaldErrorTolerance=ewaldErrorTolerance)
integrator = VVIntegrator(temperature, coll_freq, drud_temp, drud_coll_freq, dt)
#integrator = VelocityVerletIntegrator(dt)
integrator.setConstraintTolerance(constraintTolerance)
integrator.setMaxDrudeDistance(0.2*angstroms)
simulation = Simulation(topology, system, integrator, platform, platformProperties)
simulation.context.setPositions(positions)

steps = 200_000
equilibrationSteps = 150
dcdReporter = DCDReporter(f'traj/nvt_fine_{cnt}.dcd', 1)
#velReporter = VELReporter(f'traj/nvt_fine_vel_{cnt}.dcd', 1)
dataReporter = StateDataReporter(f'out/nvt_fine_{cnt}.out', 1000, totalSteps=steps,
    step=True, progress=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
    temperature=True, volume=True, density=True, separator='\t')
#checkpointReporter = CheckpointReporter('checkpoint.chk', 10000)

# Prepare the Simulation

print('Building system...')

if cnt > 1:
    with open(rst, 'r') as f:
        simulation.context.setState(XmlSerializer.deserialize(f.read()))

# Minimize and Equilibrate

if cnt == 1:
    with open(rst, 'r') as f:
        simulation.context.setState(XmlSerializer.deserialize(f.read()))
    #print('Performing energy minimization...')
    #simulation.minimizeEnergy()
    #print('Equilibrating...')
    #simulation.context.setVelocitiesToTemperature(temperature)
    #integrator.setStepSize(0.0001)
    #simulation.step(equilibrationSteps)
    simulation.currentStep = 0
    #integrator.setStepSize(dt)

# Simulate

print('Simulating...')
simulation.reporters.append(dataReporter)
#simulation.reporters.append(checkpointReporter)
simulation.reporters.append(dcdReporter)
#simulation.reporters.append(velReporter)
simulation.step(steps)


state = simulation.context.getState( getPositions=True, getVelocities=True )
with open(f'traj/nvt_fine_{cnt}.rst', 'w') as f:
    f.write(XmlSerializer.serialize(state))

#crd = simulation.context.getState(getPositions=True).getPositions()
#PDBFile.writeFile(psf.topology, crd, open('trans_equil.pdb', 'w'))
