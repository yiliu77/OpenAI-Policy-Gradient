import gym
import numpy as np
import tensorflow as tf
import time
import os

n_inputs = 8
n_hidden1 = 10
n_hidden2 = 10
n_outputs = 4

n_epochs = 100
n_games_per_epoch = 10
n_max_iter = 1700

learning_rate = 0.002
discount_rate = 0.99

save_index = 3

initializer = tf.contrib.layers.xavier_initializer()

# Creates the neural network
x = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden_layer1 = tf.layers.dense(x, n_hidden1, activation=tf.nn.relu, kernel_initializer=initializer)
hidden_layer2 = tf.layers.dense(hidden_layer1, n_hidden2, activation=tf.nn.relu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden_layer2, n_outputs, kernel_initializer=initializer)

# Softmax layer to give probability array
outputs = tf.nn.softmax(logits)
# Pick a random action based on the network's output
action = tf.multinomial(tf.log(outputs), num_samples=1)
# Pick a action without randomness (for testing not training)
testing_output = tf.argmax(outputs[0, :], 0)
y = tf.one_hot(action, n_outputs)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]

gradient_placeholders = []
grads_and_vars_feed = []
for grad, var in grads_and_vars:
    gradient_pl = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_pl)
    grads_and_vars_feed.append((gradient_pl, var))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

saver = tf.train.Saver()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

checkpoint = "./run-" + str(save_index) + ".ckpt"
if os.path.isfile(checkpoint + ".meta"):
    saver.restore(sess, checkpoint)
else:
    raise Exception("Session data not found!!")


def discount_rewards(all_rewards_t, discount_t):
    all_discounted_rewards = []
    for single_game_rewards in all_rewards_t:
        discounted_rewards = []
        accumulated_reward = 0
        for step in reversed(range(len(single_game_rewards))):
            accumulated_reward = single_game_rewards[step] + accumulated_reward * discount_t
            discounted_rewards.insert(0, accumulated_reward)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        all_discounted_rewards.append(discounted_rewards)

    return all_discounted_rewards


env = gym.make('LunarLander-v2')
env._max_episode_steps = n_max_iter

see_in_action = input("See AI in action (yes / no) ") == "yes"
if see_in_action:
    while True:
        obs = env.reset()

        for _ in range(n_max_iter):
            env.render()
            action_val = sess.run([testing_output], feed_dict={x: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0])

            print(obs)
            print(action_val[0])

            time.sleep(0.01)

            if done:
                print("done")
                break

root_logdir = "tf_logs/"
logdir = "{}/run-{}/".format(root_logdir, save_index + 1)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

for epoch in range(n_epochs):
    tick_count = 0
    reward_count = 0
    landing_count = 0
    main_thruster_count = 0
    side_thruster_count = 0

    all_rewards = []
    all_gradients = []

    for game in range(n_games_per_epoch):
        current_rewards = []
        current_gradients = []
        obs = env.reset()

        while True:
            action_val, gradients_val = sess.run([action, gradients], feed_dict={x: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])

            if action_val[0][0] == 1 or action_val[0][0] == 3:
                side_thruster_count += 1
                reward -= 0.05
            elif action_val[0][0] == 2:
                main_thruster_count += 1

            reward_count += reward
            tick_count += 1

            current_gradients.append(gradients_val)
            current_rewards.append(reward)

            if done:
                landed = True
                if reward == 100:
                    landing_count += 1
                break

        all_rewards.append(current_rewards)
        all_gradients.append(current_gradients)

    print("Epoch: " + str(epoch) + " || Avg Count: " + str(tick_count / n_games_per_epoch) + " || Avg Reward: " +
          str(reward_count / n_games_per_epoch) + " || Landing Percentage: " +
          str(100 * landing_count / n_games_per_epoch) + " || Main Thruster: " +
          str(main_thruster_count / n_games_per_epoch) + " || Side Thruster: " +
          str(side_thruster_count / n_games_per_epoch))

    summary = tf.Summary()
    summary.value.add(tag='Percent Landed', simple_value=(100 * landing_count / n_games_per_epoch))
    summary.value.add(tag='Avg Ticks', simple_value=(tick_count / n_games_per_epoch))
    summary.value.add(tag='Avg Award', simple_value=(reward_count / n_games_per_epoch))
    summary.value.add(tag='Main Thruster', simple_value=(main_thruster_count / n_games_per_epoch))
    summary.value.add(tag='Side Thruster', simple_value=(side_thruster_count / n_games_per_epoch))
    file_writer.add_summary(summary, epoch)

    # compute gradients
    computed_rewards = discount_rewards(all_rewards, discount_t=discount_rate)
    feed_dict = {}
    for var_index, gradient_pl in enumerate(gradient_placeholders):
        mean_gradient = np.mean([game_rewards * all_gradients[epoch_index][game_index][var_index]
                                 for epoch_index, epoch_rewards in enumerate(computed_rewards)
                                 for game_index, game_rewards in enumerate(epoch_rewards)], axis=0)
        feed_dict[gradient_pl] = mean_gradient
    sess.run(training_op, feed_dict=feed_dict)


env.close()
saver.save(sess, "./run-" + str(save_index + 1) + ".ckpt")
file_writer.flush()
file_writer.close()
