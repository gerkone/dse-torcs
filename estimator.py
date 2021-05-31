import numpy as np
import argparse
import os
import cv2
import h5py

class Estimator:
    def __init__(self, dataset_dir, testset_dir, load_old, resnet, track, batch, epochs, rgb):
        from network import build_model, build_model_resnet, load_model, limit_memory, get_callback
        # limit_memory()

        self.dataset_files = []
        self.evaluation_files = []

        for file in os.listdir(dataset_dir):
            if ".h5" in file:
                if track in file:
                    self.dataset_files.append(os.path.join(dataset_dir, file))

        for file in os.listdir(testset_dir):
            if ".h5" in file:
                if track in file:
                    self.evaluation_files.append(os.path.join(testset_dir, file))

        self.resnet = resnet
        self.img_height = 120
        self.img_width = 160
        self.output_size = 21

        self.rgb = rgb

        self.tbCallBack = get_callback()

        self.batch = batch

        self.epochs = epochs

        if resnet == True:
            if load_old == True:
                self.model = load_model("dse_model_resnet")
                print("Loaded saved model")
            else:
                self.model = build_model_resnet(self.img_height, self.img_width, self.output_size, self.rgb)
                print("ResNet model")
        else:
            if load_old == True:
                self.model = load_model("dse_model_plain")
                print("Loaded saved model")
            else:
                self.model = build_model(self.img_height, self.img_width, self.output_size, self.rgb)
                print("Standard model")
        self.model.summary()

    def _prepare_data(self, filename):
        dataset = h5py.File(filename, "r")
        images = np.array(dataset.get("img"))
        sensors = np.array(dataset.get("sensors"))

        # skip some steps ( shared mmeory bug )
        images = images[4:]
        sensors = sensors[4:]

        l = min(images.shape[0], sensors.shape[0])

        crop_in = np.abs(images.shape[0] - l)
        if crop_in > 0:
            images = images[:-crop_in]

        crop_out = np.abs(sensors.shape[0] - l)
        if crop_out > 0:
            sensors = sensors[:-crop_out]

        ch = (3 if self.rgb == True else 1)
        images = images.reshape((images.shape[0], self.img_height, self.img_width, ch)).astype(np.uint8)

        for i in range(images.shape[0]):
            if self.rgb == False:
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        dataset.close()

        return images, sensors


    def run(self):
        for i, ep_file in zip(range(len(self.dataset_files)), self.dataset_files):
            print("loading episode data: {} - {}/{}".format(ep_file, i + 1, len(self.dataset_files)))

            # normalize array length and convert to grayscale
            images, sensors = self._prepare_data(ep_file)

            self.train(images, sensors)

            del images
            del sensors

    def evaluate(self):
        from sklearn.decomposition import PCA
        from sklearn.metrics import explained_variance_score, r2_score
        for i, ep_file in zip(range(len(self.evaluation_files)), self.evaluation_files):
            print("loading episode data: {} - {}/{}".format(ep_file, i + 1, len(self.evaluation_files)))

            # normalize array length and convert to grayscale
            images, sensors = self._prepare_data(ep_file)

            mean_r2 = 0
            mean_evs = 0
            from PIL import Image
            for i in range(sensors.shape[0]):
                # if i > 400:
                #     img = Image.fromarray(frames[i])
                #     img.show()
                #     print(sensors[i])
                #     input()
                frame_t = np.expand_dims(images[i], axis = 0)
                pred = self.model.predict(frame_t)[0]
                # R2
                correlation_matrix = np.corrcoef(sensors[i], pred)
                correlation_xy = correlation_matrix[0,1]
                mean_r2 += correlation_xy**2
                # explained variance score
                mean_evs += explained_variance_score(sensors[i], pred)
                # TODO PCA

            self.model.evaluate(images, sensors, batch_size = self.batch, verbose = 2)

            r2 = mean_r2 / (sensors.shape[0])
            evs = mean_evs / (sensors.shape[0])
            print("R2: {:.4f} - explained variance score: {:.4f} ".format(r2, evs))

            del images
            del sensors


    def train(self, input, output):
        self.model.fit(input, output, epochs = self.epochs, batch_size = self.batch, verbose = 2)
        print("Saving model")
        name = "dse_model_plain"

        name = ("dse_model_resnet" if self.resnet == True else "dse_model_plain")
        self.model.save(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="estimate torcs sensor from image data")
    parser.add_argument("--dataset-dir", dest = "dataset_dir", help="path to the dataset directory", default="dataset/", type=str)
    parser.add_argument("--testset-dir", dest = "testset_dir", help="path to the testset directory", default="testset/", type=str)
    parser.add_argument("--load", dest = "load_old", help="set to load saved model", default="False", action="store_true")
    parser.add_argument("--resnet", dest = "resnet", help="set use resnet model", default="False", action="store_true")
    parser.add_argument("--batch", dest = "batch", help="batch size", type = int, default = 32)
    parser.add_argument("--epochs", dest = "epochs", help="n of epochs", type = int, default = 20)
    parser.add_argument("--eval", dest = "eval", help="evaluation only", default="False", action="store_true")
    parser.add_argument("--rgb", dest = "rgb", help="use rgb images (working with CNN only)", default="False", action="store_true")
    parser.add_argument("--track", dest = "track", help="train only on data from track (do not use '-' in name)", default="", type=str)

    args = parser.parse_args()

    dataset_files = []

    est = Estimator(args.dataset_dir, args.testset_dir, args.load_old, args.resnet, args.track, args.batch, args.epochs, args.rgb)

    if not args.eval == True:
        print("training")
        est.run()
    print("evaluating network")
    est.evaluate()
