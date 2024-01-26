import argparse
import time
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from utils import helpers, networks

def train_gan(opt):
    # Extract options from argparse namespace
    data_dir = opt.data_dir
    batch_size = opt.batchSize
    epochs = opt.epochs
    lambda_cyc = opt.lambda_cyc
    lambda_idt = opt.lambda_idt
    save_epoch_freq = opt.save_epoch_freq
    num_resnet_blocks = opt.num_resnet_blocks

    # Load training images
    trainA, trainB = helpers.load_train_images(data_dir)
    
    # Define the Adam optimizer for training
    train_optimizer = Adam(0.0002, 0.5)

    # Define discriminator models
    discA = networks.define_discriminator_network()
    discB = networks.define_discriminator_network()

    # Print discriminator summary
    print(discA.summary())

    # Compile the discriminators with MSE loss
    discA.compile(loss='mse', optimizer=train_optimizer, metrics=['accuracy'])
    discB.compile(loss='mse', optimizer=train_optimizer, metrics=['accuracy'])

    # Create labels for real and fake samples
    real_labels = np.ones((batch_size, 7, 7, 1))
    fake_labels = np.zeros((batch_size, 7, 7, 1))

    # Define generator models
    genA2B = networks.define_generator_network(num_resnet_blocks=num_resnet_blocks)
    genB2A = networks.define_generator_network(num_resnet_blocks=num_resnet_blocks)

    # Print generator summary
    print(genA2B.summary())

    # Make discriminators non-trainable in the adversarial model
    discA.trainable = False
    discB.trainable = False

    # Define the adversarial model
    gan_model = networks.define_adversarial_model(genA2B, genB2A, discA, discB, train_optimizer, lambda_cyc=lambda_cyc, lambda_idt=lambda_idt)

    # Setup TensorBoard for visualization
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), write_images=True, write_grads=True, write_graph=True)
    tensorboard.set_model(genA2B)
    tensorboard.set_model(genB2A)
    tensorboard.set_model(discA)
    tensorboard.set_model(discB)

    # Print configuration information
    print("Batch Size: {}".format(batch_size))
    print("Num of ResNet Blocks: {}".format(num_resnet_blocks))
    print("Starting training for {} epochs with lambda_cyc = {}, lambda_idt = {}, num_resnet_blocks = {}".format(epochs, lambda_cyc, lambda_idt, num_resnet_blocks))

    # Start training loop
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        start_time = time.time()

        dis_losses = []
        gen_losses = []

        num_batches = min(trainA.shape[0], trainB.shape[0]) // batch_size
        print("Number of batches: {} in each epoch".format(num_batches))

        # Iterate over batches
        for index in range(num_batches):
            print("Batch: {}".format(index))

            # Sample images
            realA = trainA[index * batch_size:(index + 1) * batch_size]
            realB = trainB[index * batch_size:(index + 1) * batch_size]

            # Translate images to opposite domain
            fakeB = genA2B.predict(realA)
            fakeA = genB2A.predict(realB)

            # Train the discriminator A on real and fake images
            dLossA_real = discA.train_on_batch(realA, real_labels)
            dLossA_fake = discA.train_on_batch(fakeA, fake_labels)

            # Train the discriminator B on real and fake images
            dLossB_real = discB.train_on_batch(realB, real_labels)
            dLossB_fake = discB.train_on_batch(fakeB, fake_labels)

            # Calculate the total discriminator loss
            mean_disc_loss = 0.5 * np.add(0.5 * np.add(dLossA_real, dLossA_fake), 0.5 * np.add(dLossB_real, dLossB_fake))

            print("Total Discriminator Loss: {}".format(mean_disc_loss))

            # Train the generator networks
            g_loss = gan_model.train_on_batch([realA, realB], [real_labels, real_labels, realA, realB, realA, realB])

            print("Adversarial Model losses: {}".format(g_loss))

            dis_losses.append(mean_disc_loss)
            gen_losses.append(g_loss)

        # Save losses to TensorBoard for that epoch
        helpers.save_losses_tensorboard(tensorboard, 'discriminatorA_loss', np.mean(0.5 * np.add(dLossA_real, dLossA_fake)), epoch)
        helpers.save_losses_tensorboard(tensorboard, 'discriminatorB_loss', np.mean(0.5 * np.add(dLossB_real, dLossB_fake)), epoch)
        helpers.save_losses_tensorboard(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
        helpers.save_losses_tensorboard(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

        # Save model weights and test results at specified intervals
        if epoch % save_epoch_freq == 0:
            testA, testB = helpers.load_test_images(data_dir=data_dir, num_images=2)
            fakeB = genA2B.predict(testA)
            fakeA = genB2A.predict(testB)
            reconsA = genB2A.predict(fakeB)
            reconsB = genA2B.predict(fakeA)
            identityA = genB2A.predict(testA)
            identityB = genA2B.predict(testB)

            genA2B.save('generatorAToB_temp_{}.h5'.format(epoch))
            genB2A.save('generatorBToA_temp_{}.h5'.format(epoch))
            discA.save('discriminatorA_temp_{}.h5'.format(epoch))
            discB.save('discriminatorB_temp_{}.h5'.format(epoch))
            helpers.save_test_results(testA, testB, fakeA, fakeB, reconsA, reconsA, identityA, identityB)

        print("--- {} seconds --- for epoch".format(time.time() - start_time))

    print("Training completed. Saving weights.")
    genA2B.save('generatorAToB.h5')
    genB2A.save('generatorBToA.h5')
    discA.save('discriminatorA.h5')
    discB.save('discriminatorB.h5')

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchSize', type=int, default=1, help='Batch Size to be used for training')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
