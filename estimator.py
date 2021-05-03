import numpy as np
import argparse
import os
import cv2

class Estimator:
    def __init__(self, dataset_dir, load_old, resnet):
        from network import build_model, build_model_resnet, load_model, limit_memory
        limit_memory()
        self.dataset_files = []
        for file in os.listdir(dataset_dir):
            if ".npy" in file:
                self.dataset_files.append(os.path.join(dataset_dir, file))

        self.resnet = resnet
        self.stack_depth = 3
        self.img_height = 96
        self.img_width = 128
        self.output_size = 24
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

        for i, ep_file in zip(range(len(self.dataset_files) - 1), self.dataset_files[:-1]):
            print("loading episode data: {} - {}/{}".format(ep_file, i, len(self.dataset_files)))
            episode = np.load(ep_file, allow_pickle = True)

            episode = episode[episode != np.array(None)]
            input = np.zeros((len(episode), self.img_height, self.img_width, self.stack_depth))
            output = np.zeros((len(episode), self.output_size))
            for i in range(len(episode) - self.stack_depth):
                if None not in episode[i:i + self.stack_depth]:
                    output[i] = episode[i][0]
                    input[i] = np.array([cv2.cvtColor(step[1], cv2.COLOR_BGR2GRAY) for step in episode[i:i + self.stack_depth]]).astype(np.uint8).reshape((self.img_height, self.img_width, self.stack_depth))

            self.train(input, output)

            del episode
            del input
            del output

        ep_file = self.dataset_files[-1]
        episode = np.load(ep_file, allow_pickle = True)
        input = np.zeros((len(episode), self.img_height, self.img_width, self.stack_depth))
        output = np.zeros((len(episode), self.output_size))
        for i in range(len(episode) - self.stack_depth):
            if None not in episode[i:i + self.stack_depth]:
                output[i] = episode[i][0]
                input[i] = np.array([cv2.cvtColor(step[1], cv2.COLOR_BGR2GRAY) for step in episode[i:i + self.stack_depth]]).astype(np.uint8).reshape((self.img_height, self.img_width, self.stack_depth))

        self.eval(input, output)

        del episode
        del input
        del output

    def train(self, input, output):
        print("Training")
        self.model.fit(input, output, epochs = 30, batch_size = 8)
        print("Saving model")
        name = ("dse_model_plain" if self.resnet else "dse_model_resnet")
        self.model.save(name)

    def eval(self, input, output):
        _, accuracy = self.model.evaluate(input, output)
        print("accuracy: {}%".format(accuracy * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="estimate torcs sensor from image data")
    parser.add_argument("--dataset-dir", dest = "dataset_dir", help="path to the dataset directory", default="dataset/", type=str)
    parser.add_argument("--load-old", dest = "load_old", help="set to load saved model", default="False", action="store_true")
    parser.add_argument("--resnet", dest = "resnet", help="set use resnet model", default="False", action="store_true")

    args = parser.parse_args()

    dataset_files = []

    est = Estimator(args.dataset_dir, args.load_old, args.resnet)

    est.run()
