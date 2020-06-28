from easydict import EasyDict
import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self, easydict=False):
        self.opt = None
        self.easydict = easydict
        if easydict:
            self.opt = EasyDict()
        else:
            self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self, name='label2city', gpu_ids='0', checkpoints_dir='./checkpoints',model='pix2pixHD',
                         norm = 'instance', use_dropout = None, data_type = 32, verbose = False,
                         batchSize = 1, loadSize = 512, fineSize = 512, label_nc = 20, input_nc = 3, output_nc = 3,
                         dataroot = '../Data_preprocessing/', resize_or_crop = 'scale_width', serial_batches = None, 
                         no_flip = None, nThreads = 2, max_dataset_size = float('inf'),
                         display_winsize = 512, tf_log = None, netG = 'global', ngf = 64, 
                         n_downsample_global = 4, n_blocks_global = 4, n_blocks_local = 3, 
                         n_local_enhancers = 1, niter_fix_global = 0):    
        if self.easydict:
            self.opt.name = name 
            self.opt.gpu_ids = gpu_ids
            self.opt.checkpoints_dir = checkpoints_dir
            self.opt.model = model
            self.opt.norm = norm 
            self.opt.use_dropout = use_dropout
            self.opt.data_type = data_type
            self.opt.verbose = verbose
            self.opt.batchSize = batchSize 
            self.opt.loadSize = loadSize 
            self.opt.fineSize = fineSize 
            self.opt.label_nc = label_nc 
            self.opt.input_nc = input_nc 
            self.opt.output_nc = output_nc
            self.opt.dataroot = dataroot 
            self.opt.resize_or_crop = resize_or_crop 
            self.opt.serial_batches = serial_batches 
            self.opt.no_flip = no_flip 
            self.opt.nThreads = nThreads 
            self.opt.max_dataset_size = max_dataset_size
            self.opt.display_winsize = display_winsize 
            self.opt.tf_log = tf_log 
            self.opt.netG = netG 
            self.opt.ngf = ngf 
            self.opt.n_downsample_global = n_downsample_global 
            self.opt.n_blocks_global = n_blocks_global
            self.opt.n_blocks_local = n_blocks_local 
            self.opt.n_local_enhancers = n_local_enhancers
            self.opt.niter_fix_global = niter_fix_global
        else:
        # experiment specifics
            self.parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')        
            self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
            self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
            self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
            self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
            self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
            self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
            self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

            # input/output sizes       
            self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
            self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
            self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
            self.parser.add_argument('--label_nc', type=int, default=20, help='# of input label channels')
            self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
            self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

            # for setting inputs
            self.parser.add_argument('--dataroot', type=str, default='../Data_preprocessing/') 
            self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
            self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
            self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
            self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
            self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

            # for displays
            self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
            self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

            # for generator
            self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
            self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
            self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
            self.parser.add_argument('--n_blocks_global', type=int, default=4, help='number of residual blocks in the global generator network')
            self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
            self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
            self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        if not self.easydict:
            self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
