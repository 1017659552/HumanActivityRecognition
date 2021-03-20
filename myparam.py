class Param(object):
    def dataset_dir(self):
        root_dir = 'D:\\SWUFEthesis\\data\\KTH'
        process_dir = 'D:\\SWUFEthesis\\data\\KTH_preprocess_v3'
        # process_dir = '/home/mist/KTH_preprocess'
        return root_dir,process_dir

    def img_size(self):
        crop_size = 120
        frame_width = 160
        frame_height = 120
        return crop_size,frame_width,frame_height

    def module_param(self):
        labels = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
        n_epochs = 30
        n_batch_size = 32
        n_lr = 1e-2
        return labels,n_epochs,n_batch_size,n_lr