import numpy as np
import argparse
import os
import cv2
import h5py

class Estimator:
    def __init__(self, dataset_dir, testset_dir, load_old, resnet, track):
        from network import build_model, build_model_resnet, load_model, limit_memory
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
        self.stack_depth = 3
        self.img_height = 240
        self.img_width = 320
        self.output_size = 21

        if resnet == True:
            if load_old == True:
                self.model = load_model("dse_model_resnet")
                print("Loaded saved model")
            else:
                self.model = build_model_resnet(self.stack_depth, self.img_height, self.img_width, self.output_size)
                print("ResNet model")
        else:
            if load_old == True:
                self.model = load_model("dse_model_plain")
                print("Loaded saved model")
            else:
                self.model = build_model(self.stack_depth, self.img_height, self.img_width, self.output_size)
                print("Standard model")

    def run(self):
        for i, ep_file in zip(range(len(self.dataset_files) - 1), self.dataset_files):
            print("loading episode data: {} - {}/{}".format(ep_file, i + 1, len(self.dataset_files)))

            dataset = h5py.File(ep_file, "r")
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

            frames = np.empty(shape = (images.shape[0], self.img_height, self.img_width, self.stack_depth), dtype = np.uint8)

            from PIL import Image
            for i in range(images.shape[0] - self.stack_depth):
                frames[i] = np.stack([cv2.cvtColor(step, cv2.COLOR_BGR2GRAY) for step in images[i:i + self.stack_depth]], axis = -1).astype(np.uint8)
                # if i > 300:
                #     print(sensors[i])
                #     img = Image.fromarray(frames[i][..., 0])
                #     img.show()
                #     input()

            # bad way
            frames = frames[:-self.stack_depth]
            sensors = sensors[:-self.stack_depth]

            self.train(frames, sensors)

            del images
            del sensors

            dataset.close()

    def evaluate(self):
        for i, ep_file in zip(range(len(self.evaluation_files)), self.evaluation_files):
            print("loading episode data: {} - {}/{}".format(ep_file, i + 1, len(self.evaluation_files)))

            dataset = h5py.File(ep_file, "r")
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

            frames = np.empty(shape = (images.shape[0], self.img_height, self.img_width, self.stack_depth), dtype = np.uint8)

            for i in range(images.shape[0] - self.stack_depth):
                frames[i] = np.stack([cv2.cvtColor(step, cv2.COLOR_BGR2GRAY) for step in images[i:i + self.stack_depth]], axis = -1).astype(np.uint8)

            # bad way
            frames = frames[:-self.stack_depth]
            sensors = sensors[:-self.stack_depth]
            # for i in range(len(frames)):
            #     frame_t = np.expand_dims(frames[i], axis = 0)
            #
            #     pred = self.model.predict(frame_t)[0]
            #     print(pred)
            #     print(sensors[i])
            #     print(((pred - sensors[i])**2).mean(axis = None))
            #     print()

            _, accuracy = self.model.evaluate(frames, sensors)
            print("accuracy: {:.2f}%".format(accuracy * 100))

            del images
            del sensors


    def train(self, input, output):
        self.model.fit(input, output, epochs = 15, batch_size = 8)
        print("Saving model")
        name = ("dse_model_resnet" if self.resnet else "dse_model_plain")
        self.model.save(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="estimate torcs sensor from image data")
    parser.add_argument("--dataset-dir", dest = "dataset_dir", help="path to the dataset directory", default="dataset/", type=str)
    parser.add_argument("--testset-dir", dest = "testset_dir", help="path to the testset directory", default="testset/", type=str)
    parser.add_argument("--load", dest = "load_old", help="set to load saved model", default="False", action="store_true")
    parser.add_argument("--resnet", dest = "resnet", help="set use resnet model", default="False", action="store_true")
    # parser.add_argument("--no-eval", dest = "no_eval", help="do not perform tests", default="False", action="store_true")
    # parser.add_argument("--no-train", dest = "no_train", help="do not perform training", default="False", action="store_true")
    parser.add_argument("--track", dest = "track", help="train only on data from track (no '-' in name)", default="", type=str)

    args = parser.parse_args()

    dataset_files = []

    est = Estimator(args.dataset_dir, args.testset_dir, args.load_old, args.resnet, args.track)

    print("training")
    est.run()
    print("evaluating network")
    est.evaluate()
