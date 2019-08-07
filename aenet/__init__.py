import cPickle
import logging
import ntpath
import numpy as np
import os
import sys
import tempfile

from keras.models import Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Input, Flatten


#import lasagne
#import theano
#from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers import InputLayer, MaxPool2DLayer, DenseLayer
from moviepy.editor import VideoFileClip

from . import htkfio

logger = logging.getLogger('aenet')
logger.setLevel(logging.INFO)
lh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
lh.setFormatter(formatter)
logger.addHandler(lh)

feat_shape = [3, 200, 50]  # channel, time. frequency

if 'AENET_DATA_DIR' in os.environ:
    AENET_DATA_DIR = os.environ['AENET_DATA_DIR']
else:
    logger.error("Environment variable AENET_DATA_DIR not set. Set to current directory")
    AENET_DATA_DIR = './'


class AENet:
    def __init__(self,
                 weight_file='%s/model.pkl' % AENET_DATA_DIR,
                 feat_mean=None, feat_std=None,
                 layer='fc6',
                 BATCH_SIZE = 10,
                 HTK_ROOT=AENET_DATA_DIR):  # Directory containing HTK files

        self.BATCH_SIZE = BATCH_SIZE

        # Build model
        self.model = self.build_keras_model()
        print('Finised building model')
        
        # Set the pre-trained weights (takes some time)
        #self.set_weights(net['prob'], weight_file)

        # Compile prediction function
        #prediction = lasagne.layers.get_output(net[layer], deterministic=True)
        #self.pred_fn = train_on_batch(x, y,) theano.function([net['input'].input_var], prediction, allow_input_downcast=True)

        if feat_mean is None:
            self.feat_mean = np.load('%s/gmean.npy' % AENET_DATA_DIR)
        else:
            self.feat_mean = feat_mean

        if feat_std is None:
            self.feat_std = np.load('%s/gstd.npy' % AENET_DATA_DIR)
        else:
            self.feat_std = feat_std

        # Define HTK file locations
        self.HCopyExe = '%s/HCopy' % HTK_ROOT
        self.HConfigFile = '%s/configmfb.hcopy' % HTK_ROOT


    def build_keras_model(self):
        print(feat_shape)
        inputs = Input(shape=(feat_shape[0], feat_shape[1], feat_shape[2]))
        # ----------- 1st layer group ---------------
        x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu')(inputs)
        x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(1, 2))(x)
        # ----------- fully connected layer group ---------------
        x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same',activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # ----------- fully connected layer group ---------------
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(41, activation="relu")(x)
        predictions = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
        print(model.summary())

        return model
    def set_weights(self, net, model_file):
        '''
        Sets the parameters of the model using the weights stored in model_file
        Parameters
        ----------
        net: a Lasagne layer

        model_file: string
            path to the model that containes the weights

        Returns
        -------
        None

        '''
        with open(model_file) as f:
            logger.info('Load pretrained weights from %s ...' % model_file)
            model = cPickle.load(f)
        logger.info('Set the weights...')
        #lasagne.layers.set_all_param_values(net, model['param_values'], trainable=True)

    def write_wav(self, video_obj, target_wav_file):
        '''
        Writes the audio stream of a video as a wav suitable as input to HTK

        ----------
        video_obj: a moviepy VideoFileClip

        target_wav_file: path to write the wav file to

        Returns
        -------
        None

        '''
        assert isinstance(video_obj, VideoFileClip), "video needs to be a instance of VideoFileClip"

        # Write audio stream of video to file in the desired format
        video_obj.audio.write_audiofile(target_wav_file, fps=16000,  # Set fps to 16k
                                        codec='pcm_s16le',
                                        ffmpeg_params=['-ac', '1'])  # Convert to mono

    def extract_fbank_htk(self, scriptfile):
        '''
        :param scriptfile: path to the HCopy's script file
        :return: list of path to feature files
        '''

        with open(scriptfile, 'r') as scpf:
            featfiles = scpf.readlines()
        featfiles = [f.split(' ')[1].replace('\n', '') for f in featfiles]

        for f in featfiles:
            if not os.path.exists(ntpath.dirname(f)):
                os.makedirs(ntpath.dirname(f))

        cmd = self.HCopyExe + ' -C ' + self.HConfigFile + ' -S ' + scriptfile
        os.system(cmd)

        return featfiles

    def slice_data_gen(self, feat=None, datalen=100, outdatalen=160, slide=80, fdim=150):
        '''
        This function return slices of input feat slide
        :param feat: test featur vector
        :param datalen: length of test data
        :param outdatalen: length of output data
        :param slide:
        :param fdim:
        :return:
        '''
        outsampleN = int((datalen - outdatalen) / slide + 1)

        td = np.empty((outsampleN, fdim * outdatalen), dtype=np.float32)
        data = feat.reshape(datalen, fdim)

        for i in range(outsampleN):
            tmpfeat = data[i * slide:i * slide + outdatalen]
            td[i] = tmpfeat.reshape(-1)

        return td

    def get_feat_sequence(self, htkfile, shift):
        '''
        :param htkfile: a feature file path
        :param slide: a window slide step. Normally 50% of input patch size.
        :param feat_mean: mean vector of features
        :param feat_std:  standard deviation vector of features
        :return: feature sequence
        '''
        feat = htkfio.open(htkfile, 'rb').getall()
        datalen = feat.shape[0]
        input_shape = [-1] + feat_shape

        # normalizing
        feat = (feat - self.feat_mean[None, :]) / self.feat_std[None, :]

        # if data length is shorter than target output length, repeat signal.
        if datalen < input_shape[2]:
            while datalen < input_shape[2]:
                cpylen = datalen
                feat = np.vstack((feat, feat[:cpylen]))  # repeat
                datalen = feat.shape[0]

        feats = self.slice_data_gen(feat, datalen, input_shape[2], shift, input_shape[1] * input_shape[3])
        feats = feats.reshape([input_shape[0], input_shape[2], input_shape[1], input_shape[3]])

        return np.swapaxes(feats, 1, 2)

    def train(self, wavfilelist, labels, shift=100):
        '''
        :param wavfilelist: list of wave files, 16kHz, 16bit, mono
        :param shift: number of frames to shift input patch. 1 frame = 10 msec in 16kHz
        :return: list of sequences of AENet features
        '''

        tmp_dir = tempfile.mkdtemp()
        tmp_file = tempfile.mktemp(suffix='scp')

        mfb_files = []
        with open(tmp_file, 'w') as f:

            for wf in wavfilelist:
                wf = wf.rstrip()
                mfb_files.append('%s/%s' % (tmp_dir, ntpath.basename(wf).replace('.wav', '.mfb')))
                f.write(wf + ' ' + mfb_files[-1] + '\n')

        # extract mel filter bank output
        self.extract_fbank_htk(scriptfile=tmp_file)
        os.remove(tmp_file)

        training_data = None
        labels = None

        aenet_feat = []
        for f in mfb_files:
            if os.path.exists(f):
                print(f)
                mfb = self.get_feat_sequence(htkfile=f, shift=shift)
                if training_data is None:
                    training_data = mfb
                else:
                    training_data = np.append(training_data, mfb, axis=0)
                if '0.mfb' in f:
                    label_val = 0.0
                else:
                    label_val = 1.0
                labels_curr = np.full(mfb.shape[0], label_val)
                if labels is None:
                    labels = labels_curr
                else:
                    labels=np.append(labels, labels_curr, axis=0)

        print(training_data.shape)
        print(labels.shape)
        hist=self.model.fit(training_data, labels)
        print(hist.history)
