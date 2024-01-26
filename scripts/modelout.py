import argparse
from utils import helpers, networks

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchSize', type=int, default=1, help='Batch Size to be used for training')
    parser.add_argument('--data_dir', type=str, default='data/male_female/', help='Directory where train and test images are present')

    # Parse command line arguments and store them in opt
    opt, _ = parser.parse_known_args()

    # Extract options
    data_dir = opt.data_dir
    batch_size = opt.batchSize

    print("Data Directory: {}".format(data_dir))
    print("Batch Size: {}".format(batch_size))

    # Define and load generator models
    genA2B = networks.define_generator_network()
    genB2A = networks.define_generator_network()
    genA2B.load_weights("generatorAToB.h5")
    genB2A.load_weights("generatorBToA.h5")

    # Load test images
    testA, testB = helpers.load_test_images(data_dir=data_dir, num_images=batch_size)

    # Generate images
    fakeB = genA2B.predict(testA)
    fakeA = genB2A.predict(testB)

    # Get reconstructed images
    reconsA = genB2A.predict(fakeB)
    reconsB = genA2B.predict(fakeA)

    # Get identity-transformed images
    identityA = genB2A.predict(testA)
    identityB = genA2B.predict(testB)

    # Save test results
    helpers.save_test_results(testA, testB, fakeA, fakeB, reconsA, reconsB, identityA, identityB)

if __name__ == '__main__':
    main()
