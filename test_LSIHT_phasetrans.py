__author__ = 'Nic'

import numpy
import scipy.io
import datetime
from collections import namedtuple

import sys, socket
hostname = socket.gethostname()
if hostname == 'caraiman':
    pyCSalgos_path = '/home/nic/code/pyCSalgos'
elif hostname == 'nclejupc':
    pyCSalgos_path = '/home/ncleju/code/pyCSalgos'
elif hostname == 'nclejupchp':
    pyCSalgos_path = '/home/ncleju/Work/code/pyCSalgos'
#sys.path.insert(0,'/home/ncleju/code/pyCSalgos')
#sys.path.append('D:\\Facultate\\Code\\pyCSalgos')
sys.path.append(pyCSalgos_path)

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

#from pyCSalgos import AnalysisPhaseTransition
#from pyCSalgos import UnconstrainedAnalysisPursuit
#from pyCSalgos import AnalysisBySynthesis
#from pyCSalgos import OrthogonalMatchingPursuit
from LSIHT import LeastSquaresIHT
from pyCSalgos import SynthesisPhaseTransition
from pyCSalgos import IterativeHardThresholding
from pyCSalgos import OrthogonalMatchingPursuit
from pyCSalgos import ApproximateMessagePassing

# Named tuple to define parameters
Params = namedtuple('Params',
    ['name', 'signal_size', 'dict_size', 'deltas', 'rhos', 'snr_db', 'num_data', 'solvers', 'solver_names', 'success_thresh', 'save_folder', 'dictionary', 'acqumatrix'])
# Dictionary for parameter sets
p = dict()

#==============================================================================
# PARAMETER SETS
#==============================================================================
# Select the parameter set actually used in the main() function below
#
name    = 'fig_exact_LSIHT_test1'               # Name 
signal_size, dict_size = 100, 120               # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.1)      # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.1)      # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 2                              # Number of signals to average 
solvers        = [LeastSquaresIHT(1, 10000), IterativeHardThresholding(0.65, 1e-7, 1e-16, sparsity="real", maxiter=10000, debias='real')]
solver_names   = ['LSIHT','IHT']                # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = 'randn'
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)
#
name    = 'fig_exact_LSIHT_dictRandn2'          # Name 
signal_size, dict_size = 200, 240               # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 20                             # Number of signals to average 
solvers        = [LeastSquaresIHT(1, 10000), IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=10000, debias='real'), OrthogonalMatchingPursuit(1e-7), ApproximateMessagePassing(1e-7, 1000)]
solver_names   = ['LSIHT','IHT', 'OMP', 'AMP']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = 'randn'
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)
#
name    = 'fig_exact_LSIHT_dictRandnSquare'     # Name 
signal_size, dict_size = 200, 200               # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 20                             # Number of signals to average 
solvers        = [LeastSquaresIHT(1, 10000), IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=10000, debias='real'), OrthogonalMatchingPursuit(1e-7), ApproximateMessagePassing(1e-7, 1000)]
solver_names   = ['LSIHT','IHT', 'OMP', 'AMP']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = 'randn'
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)
#
name    = 'fig_exact_LSIHT_dictRandnSquare_onlyIHT_mu0p75'     # Name 
signal_size, dict_size = 200, 200               # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 20                             # Number of signals to average 
solvers        = [IterativeHardThresholding(0.75, 1e-7, 1e-16, sparsity="real", maxiter=10000, debias='real')]
solver_names   = ['IHT']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = 'randn'
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)
#
name    = 'fig_exact_LSIHT_dictRandnSquare_onlyIHT_mu0p1'     # Name 
signal_size, dict_size = 200, 200               # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 5                              # Number of signals to average 
solvers        = [IterativeHardThresholding(0.1, 1e-7, 1e-16, sparsity="real", maxiter=10000, debias='real')]
solver_names   = ['IHT']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = 'randn'
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)
#
name    = 'fig_exact_LSIHT_dictRandnSquare_onlyIHT_adaptive'     # Name 
signal_size, dict_size = 200, 200               # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 5                             # Number of signals to average 
solvers        = [IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=10000, debias='real')]
solver_names   = ['IHT']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = 'randn'
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)
#
name    = 'fig_exact_LSIHT_DictUnitProjRandn'     # Name 
signal_size, dict_size = 200, 200               # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 100                             # Number of signals to average 
solvers        = [LeastSquaresIHT(1, 1000), IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), OrthogonalMatchingPursuit(1e-7), ApproximateMessagePassing(1e-7, 1000)]
solver_names   = ['LSIHT','IHTa', 'IHTmu1', 'OMP', 'AMP']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = numpy.eye(200)
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)

#
name    = 'fig_exact_LSIHT_DictUnitProjRandn_IHTmu1'     # Name 
signal_size, dict_size = 200, 200               # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 2                              # Number of signals to average 
solvers        = [IterativeHardThresholding(1, 1e-7, 1e-16)]
solver_names   = ['IHT']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = numpy.eye(200)
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)

#============================================
# Use a learned dictionary
learneddict = scipy.io.loadmat('dicts/test1_n256_N1024_m150_1_D.mat')['D']
for i in range(learneddict.shape[1]):
    learneddict[:,i] = learneddict[:,i] / numpy.sqrt(numpy.sum( learneddict[:,i]**2 ))  # normalize columns
#plt.plot(learneddict['D'][:,0])
#plt.show()
#[U,S,Vt] = numpy.linalg.svd(learneddict)
#plt.plot(S)
#plt.show()
#
name           = 'fig_exact_LSIHT_DictLearned256x1024ProjRandn'  # Name 
signal_size, dict_size = 256, 1024                # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 20                             # Number of signals to average 
solvers        = [LeastSquaresIHT(1, 10000), IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=10000, debias='real'), IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=10000, debias='real'), OrthogonalMatchingPursuit(1e-7), ApproximateMessagePassing(1e-7, 1000)]
solver_names   = ['LSIHT','IHTa', 'IHTmu1', 'OMP', 'AMP']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = learneddict  # Use the learned dictionary
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)

# A smaller dictionary
learneddict = scipy.io.loadmat('dicts/Dtrain_zeromean_64x256_L10.mat')['D']
for i in range(learneddict.shape[1]):
    learneddict[:,i] = learneddict[:,i] / numpy.sqrt(numpy.sum( learneddict[:,i]**2 ))  # normalize columns
#plt.plot(learneddict[:,0])
#plt.show()
#[U,S,Vt] = numpy.linalg.svd(learneddict)
#plt.plot(S)
#plt.show()
name           = 'fig_exact_LSIHT_DictLearned64x256ProjRandn'  # Name 
signal_size, dict_size = 64, 256              # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 2                             # Number of signals to average 
solvers        = [LeastSquaresIHT(1, 1000), IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), OrthogonalMatchingPursuit(1e-7), ApproximateMessagePassing(1e-7, 1000)]
solver_names   = ['LSIHT','IHTa', 'IHTmu1', 'OMP', 'AMP']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = learneddict  # Use the learned dictionary
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)

#
learneddict = scipy.io.loadmat('dicts/Dtrain_zeromean_64x80_L10.mat')['D']
for i in range(learneddict.shape[1]):
    learneddict[:,i] = learneddict[:,i] / numpy.sqrt(numpy.sum( learneddict[:,i]**2 ))  # normalize columns
#plt.plot(learneddict[:,0])
#plt.show()
#[U,S,Vt] = numpy.linalg.svd(learneddict)
#plt.plot(S)
#plt.show()
name           = 'fig_exact_LSIHT_DictLearned64x80ProjRandn'  # Name 
signal_size, dict_size = 64, 80              # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 100                             # Number of signals to average 
solvers        = [LeastSquaresIHT(1, 1000), IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), OrthogonalMatchingPursuit(1e-7), ApproximateMessagePassing(1e-7, 1000)]
solver_names   = ['LSIHT','IHTa', 'IHTmu1', 'OMP', 'AMP']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = learneddict  # Use the learned dictionary
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)

#
learneddict = scipy.io.loadmat('dicts/Dtrain_zeromean_64x80_L10.mat')['D']
for i in range(learneddict.shape[1]):
    learneddict[:,i] = learneddict[:,i] / numpy.sqrt(numpy.sum( learneddict[:,i]**2 ))  # normalize columns
#plt.plot(learneddict[:,0])
#plt.show()
#[U,S,Vt] = numpy.linalg.svd(learneddict)
#plt.plot(S)
#plt.show()
name           = 'fig_exact_LSIHT_DictLearned64x80ProjRandn_thresh1em3'  # Name 
signal_size, dict_size = 64, 80              # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 100                             # Number of signals to average 
solvers        = [LeastSquaresIHT(1, 1000), IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), OrthogonalMatchingPursuit(1e-7), ApproximateMessagePassing(1e-7, 1000)]
solver_names   = ['LSIHT','IHTa', 'IHTmu1', 'OMP', 'AMP']  # Names used for figure files
success_thresh = 1e-3                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
dictionary     = learneddict  # Use the learned dictionary
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)

#=================================================
# Use a random dictionary with decaying spectrum
name           = 'fig_exact_LSIHT_DictRandSpectExp200x200ProjRandn'  # Name 
signal_size, dict_size = 200, 200              # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 20                            # Number of signals to average 
solvers        = [LeastSquaresIHT(1, 1000), IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), OrthogonalMatchingPursuit(1e-7), ApproximateMessagePassing(1e-7, 1000)]
solver_names   = ['LSIHT','IHTa', 'IHTmu1', 'OMP', 'AMP']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
#dictionary     = learneddict  # Use the learned dictionary
numpy.random.seed(1234)
dictionary = numpy.random.randn(signal_size, dict_size)
[U,S,Vt] = numpy.linalg.svd(dictionary)
A = 5
RCconst = 0.05
S = 5 * numpy.exp(-RCconst * numpy.arange(S.size))
#plt.plot(S)
#plt.show()
dictionary = U @ numpy.diag(S) @ Vt
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)


#=================================================
# Use a random dictionary with decaying spectrum, normalized
name           = 'fig_exact_LSIHT_DictRandSpectExpNormed200x200ProjRandn'  # Name 
signal_size, dict_size = 200, 200              # Signal dimensions
deltas         = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, x axis
rhos           = numpy.arange(0.1, 1, 0.05)     # Phase transition grid, y axis
snr_db         = numpy.Inf                      # SNR ratio
num_data       = 20                            # Number of signals to average 
solvers        = [LeastSquaresIHT(1, 1000), IterativeHardThresholding(0, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), IterativeHardThresholding(1, 1e-7, 1e-16, sparsity="real", maxiter=1000, debias='real'), OrthogonalMatchingPursuit(1e-7), ApproximateMessagePassing(1e-7, 1000)]
solver_names   = ['LSIHT','IHTa', 'IHTmu1', 'OMP', 'AMP']  # Names used for figure files
success_thresh = 1e-6                           # Threshold for considering successful recovery
save_folder    = 'save'                         # Where to save
#dictionary     = learneddict  # Use the learned dictionary
numpy.random.seed(1234)
dictionary = numpy.random.randn(signal_size, dict_size)
[U,S,Vt] = numpy.linalg.svd(dictionary)
A = 5
RCconst = 0.05
S = 5 * numpy.exp(-RCconst * numpy.arange(S.size))
plt.plot(S)
#plt.show()
plt.savefig('save/' + name + '_spectrum' + '.' + 'pdf', bbox_inches='tight')
plt.savefig('save/' + name + '_spectrum' + '.' + 'png', bbox_inches='tight')
dictionary = U @ numpy.diag(S) @ Vt
for i in range(dictionary.shape[1]):
    dictionary[:,i] = dictionary[:,i] / numpy.sqrt(numpy.sum( dictionary[:,i]**2 ))  # normalize columns
acqumatrix     = "randn"
p[name] = Params(name, signal_size, dict_size, deltas, rhos, snr_db, num_data, solvers, solver_names, success_thresh, save_folder, dictionary, acqumatrix)


#============================================

def run(p):

    time_start = datetime.datetime.now()
    print(time_start.strftime("%Y-%m-%d-%H:%M:%S:%f") + " --- Started running %s..."%(p.name))

    #solvers = [LeastSquaresIHT(0.1, 10000), IHT]
    
    file_prefix = p.save_folder + '/' + p.name
    if len(p.solver_names) > 1:
        figs_filename = [file_prefix + '_' + s for s in p.solver_names]
    else:
        figs_filename = file_prefix
    
    pt = SynthesisPhaseTransition(p.signal_size, p.dict_size, p.deltas, p.rhos, p.num_data, p.snr_db, dictionary=p.dictionary, acqumatrix=p.acqumatrix)
    pt.set_solvers(p.solvers)
    pt.run(processes=1, random_state=123)
    #pt.savedata(data_filename)
    pt.savedescription(file_prefix)
    pt.plot(subplot=False, solve=True, check=False, thresh=p.success_thresh, show=False,
            basename=figs_filename, saveexts=['png', 'pdf'])
    #pt.plot_global_error(shape=((len(alphas),len(betas))), thresh=1e-6, show=False,
    #                     basename=file_prefix+'_global', saveexts=['png', 'pdf'], textfilename=file_prefix+'_global.txt')  # old comment: this order because C order (?)

    time_end = datetime.datetime.now()
    print(time_end.strftime("%Y-%m-%d-%H:%M:%S:%f") + " --- Ended. Elapsed: " + \
          str((time_end - time_start).seconds) + " seconds")


if __name__ == "__main__":
    # For profiling
    #import cProfile
    #cProfile.run('run()', 'profile')

    # Test 1:
    run(p['fig_exact_LSIHT_DictUnitProjRandn'])
    
    # Test 2:
    #run(p['fig_exact_LSIHT_DictRandSpectExpNormed200x200ProjRandn'])

    # Test 3:
    #run(p['fig_exact_LSIHT_DictLearned64x80ProjRandn'])
    #run(p['fig_exact_LSIHT_DictLearned64x80ProjRandn_thresh1em3'])


