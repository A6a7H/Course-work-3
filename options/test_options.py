from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, ntest=float("inf"), results_dir='./results/', aspect_ratio=1.0,
                                phase='test', which_epoch='latest', how_many=1000, cluster_path='features_clustered_010.npy',
                                use_encoded_image=None, export_onnx=None, engine=None, onnx=None, **kwargs):
        BaseOptions.initialize(**kwargs)
        if self.easyDict:
            self.ntest = ntest
            self.results_dir = results_dir
            self.aspect_ratio = aspect_ratio
            self.phase = phase
            self.which_epoch = which_epoch
            self.how_many = how_many
            self.cluster_path = cluster_path
            self.use_encoded_image = use_encoded_image
            self.export_onnx = export_onnx
            self.engine = engine
            self.onnx = onnx
        else:
            self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
            self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
            self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
            self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
            self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
            self.parser.add_argument('--how_many', type=int, default=1000, help='how many test images to run')       
            self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
            self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
            self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
            self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
            self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        self.isTrain = False
