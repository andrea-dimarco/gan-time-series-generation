'''
Use original data as training set to generater synthetic data (time-series)
'''

# Necessary Packages
import tensorflow as tf
import tf_slim as slim
import numpy as np
from utils import rnn_cell, random_generator, batch_generator

import torch
from sklearn.preprocessing import MinMaxScaler
from data_loading import sine_data_generation


# TODO: Make this as a class
def timegan(ori_data, parameters, scaler=MinMaxScaler()):
  '''
  TimeGAN function.
  
  Use original data as training set to generater synthetic data (time-series)
  
  Args:
    - ori_data: original time-series data
    - parameters: TimeGAN network parameters
    
  Returns:
    - generated_data: generated time-series data
  '''
  # Initialization on the Graph
  tf.compat.v1.reset_default_graph() # clears the graph for this thread

  # Basic Parameters
  no, seq_len, dim = ori_data.shape
    
  # Maximum sequence length and each sequence length
  # TODO: this must not be a list
  ori_time = (torch.ones(no) * seq_len).tolist()
  print(seq_len)
  
  # Normalization
  scaler.fit(ori_data.reshape(no, -1))
  max_val = scaler.data_max_
  min_val = scaler.data_min_
  normalized_data = torch.as_tensor(scaler.transform(ori_data.reshape(no, -1))).reshape(no, seq_len, dim)
              
  ## Build a RNN networks          
  # Network Parameters
  hidden_dim   = parameters['hidden_dim'] 
  num_layers   = parameters['num_layer']
  iterations   = parameters['iterations']
  batch_size   = parameters['batch_size']
  module_name  = parameters['module'] 
  z_dim        = dim
  gamma        = 1
    
  # Input place holders
  # TODO: as tensors
  tf.compat.v1.disable_eager_execution()
  X = tf.compat.v1.placeholder(tf.float32, [None, seq_len, dim], name = "myinput_x") # none means dinamic
  Z = tf.compat.v1.placeholder(tf.float32, [None, seq_len, z_dim], name = "myinput_z")
  T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")

    
  # Embedder & Recovery
  # Embedd X into its latent space representation H
  #  and then reconstruct X_tilde from the representation H
  #  X_tilde must be the same as X
  H = embedder(X, T, module_name=module_name, hidden_dim=hidden_dim, num_layers=num_layers)
  X_tilde = recovery(H, T, module_name=module_name, hidden_dim=hidden_dim, num_layers=num_layers)
  

  # Generator
  # Instead of generating the sequence directy from the random vector
  #  we first generate the LATENT SPACE representation of the sequence to be generated
  #  note that E_hat is a generated sequence IN the lower dimension latent space
  #  keep in mind that Z has a lower dimensionality than the latent space
  #  the generator is EXPANDING the vector Z 
  E_hat = generator(Z, T, module_name=module_name, hidden_dim=hidden_dim, num_layers=num_layers) 

  # The supervisors aim to measure the DIFFERENCES between the latent space defined by te EMBEDDER 
  #  and the latent space defined by the GENERATOR, they must span the same latent space
  H_hat = supervisor(E_hat, T, module_name=module_name, hidden_dim=hidden_dim, num_layers=num_layers) # the supervisor must tell if E_hat is fake (it is)
  H_hat_supervise = supervisor(H, T, module_name=module_name, hidden_dim=hidden_dim, num_layers=num_layers) # the supervisor must tell if H is fake (it isn't)
  
  print("__________________________________")
  print("\nLength of sequnces\n T:\n" + str(T)) 
  print("\nReal data\n X:\n" + str(X)) 
  print("\nReconstruction of X\n X_tilde:\n" + str(X_tilde)) 
  print("\nEmbedding of real data\n H:\n" + str(H)) 
  print("\nFake Embedding from Z\n E_hat:\n" + str(E_hat))
  print("\nFake Latent Space for the sequences\n H_hat:\n" + str(H_hat)) 

  # Synthetic data
  #  Fake data generated from the fake latent space obtained from Z.
  #   X_tilde is the data reconstructed from the latent space representation of X, the real data
  #   X_hat is the data reconstructed from the latent space representation of Z, the random noise
  X_hat = recovery(H_hat, T, module_name=module_name, hidden_dim=hidden_dim, num_layers=num_layers)
    
  # Discriminator
  #  discriminate between the latent space representations, not the feature reconstructions
  Y_fake   = discriminator(H_hat, T, module_name=module_name, hidden_dim=hidden_dim, num_layers=num_layers)
  Y_real   = discriminator(H, T, module_name=module_name, hidden_dim=hidden_dim, num_layers=num_layers)     
  Y_fake_e = discriminator(E_hat, T, module_name=module_name, hidden_dim=hidden_dim, num_layers=num_layers)

  return 0
'''
  # Variables        
  e_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('embedder')]
  r_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('recovery')]
  g_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('generator')]
  s_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('supervisor')]
  d_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]
    
  # Discriminator loss
  D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
  D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
  D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
  D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            
  # Generator loss
  # 1. Adversarial loss
  G_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
  G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    
  # 2. Supervised loss
  G_loss_S = tf.compat.v1.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,:-1,:])
    
  # 3. Two Momments
  G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
  G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    
  G_loss_V = G_loss_V1 + G_loss_V2
    
  # 4. Summation
  G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V 
            
  # Embedder network loss
  E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
  E_loss0 = 10*tf.sqrt(E_loss_T0)
  E_loss = E_loss0  + 0.1*G_loss_S
    
  # optimizer
  E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0, var_list = e_vars + r_vars)
  E_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)
  D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
  G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list = g_vars + s_vars)      
  GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars)   
        
  ## TimeGAN training   
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())
    
  # 1. Embedding network training
  print('Start Embedding Network Training')
    
  for itt in range(iterations):
    # Set mini-batch
    X_mb, T_mb = batch_generator(normalized_data, ori_time, batch_size)           
    # Train embedder        
    _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})        
    # Checkpoint
    if itt % 1000 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) ) 
      
  print('Finish Embedding Network Training')
    
  # 2. Training only with supervised loss
  print('Start Training with Supervised Loss Only')
    
  for itt in range(iterations):
    # Set mini-batch
    X_mb, T_mb = batch_generator(normalized_data, ori_time, batch_size)    
    # Random vector generation   
    Z_mb = random_generator(batch_size, z_dim, T_mb, seq_len)
    # Train generator       
    _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})       
    # Checkpoint
    if itt % 1000 == 0:
      print('step: '+ str(itt)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)) )
      
  print('Finish Training with Supervised Loss Only')
    
  # 3. Joint Training
  print('Start Joint Training')
  
  for itt in range(iterations):
    # Generator training (twice more than discriminator training)
    for kk in range(2):
      # Set mini-batch
      X_mb, T_mb = batch_generator(normalized_data, ori_time, batch_size)               
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, T_mb, seq_len)
      # Train generator
      _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
       # Train embedder        
      _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})   
           
    # Discriminator training        
    # Set mini-batch
    X_mb, T_mb = batch_generator(normalized_data, ori_time, batch_size)           
    # Random vector generation
    Z_mb = random_generator(batch_size, z_dim, T_mb, seq_len)
    # Check discriminator loss before updating
    check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
    # Train discriminator (only when the discriminator does not work well)
    if (check_d_loss > 0.15):        
      _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
    # Print multiple checkpoints
    if itt % 1000 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + 
            ', d_loss: ' + str(np.round(step_d_loss,4)) + 
            ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) + 
            ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) + 
            ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) + 
            ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4))  )
  print('Finish Joint Training')
    
  ## Synthetic data generation
  Z_mb = random_generator(no, z_dim, ori_time, seq_len)
  generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: normalized_data, T: ori_time})    
    
  generated_data = list()
    
  for i in range(no):
    temp = generated_data_curr[i,:ori_time[i],:]
    generated_data.append(temp)
        
  # Renormalization
  generated_data = generated_data * max_val
  generated_data = generated_data + min_val
    
  return generated_data
'''

def embedder(X, T, module_name='gru', num_layers=3, hidden_dim=24):
  '''
  Embedding network between original feature space to latent space.
  
  Args:
    - X: input time-series features
    - T: input time information
    - module_name: the string for te module to be used
    - num_layers: number of layers
    - hidden_dim: number of hidden dimensions
    
  Returns:
    - H: embeddings
  '''
  with tf.compat.v1.variable_scope("embedder", reuse = tf.compat.v1.AUTO_REUSE):
    # TODO: remake this with PyTorch ffs
    # this is similar to SEQUENTIAL
    e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
    e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
    # the activation function is the sigmoid
    H = slim.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
  return H

def recovery(H, T, module_name='gru', hidden_dim=3, num_layers=24):   
  '''
  recovery network from latent space to original space.
  
  Args:
    - H: latent representation
    - T: input time information
    
  Returns:
    - X_tilde: recovered data
  '''     
  with tf.compat.v1.variable_scope("recovery", reuse = tf.compat.v1.AUTO_REUSE):       
    r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
    r_outputs, r_last_states = tf.compat.v1.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
    X_tilde = slim.fully_connected(r_outputs, dim, activation_fn=tf.nn.sigmoid) 
  return X_tilde

def generator (Z, T, module_name='gru', hidden_dim=3, num_layers=24):  
  '''
  Generator function: Generate time-series data in latent space.
  
  Args:
    - Z: random variables
    - T: input time information
    
  Returns:
    - E: generated embedding
  '''        
  with tf.compat.v1.variable_scope("generator", reuse = tf.compat.v1.AUTO_REUSE):
    e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
    e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
    E = slim.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
  return E

def discriminator(H, T, module_name='gru', hidden_dim=3, num_layers=24):
  '''
  Discriminate the original and synthetic time-series data.
  
  Args:
    - H: latent representation
    - T: input time information
    
  Returns:
    - Y_hat: classification results between original and synthetic time-series
  '''        
  with tf.compat.v1.variable_scope("discriminator", reuse = tf.compat.v1.AUTO_REUSE):
    d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
    d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
    Y_hat = slim.fully_connected(d_outputs, 1, activation_fn=None) 
  return Y_hat 

def supervisor(H, T, module_name='gru', hidden_dim=3, num_layers=24): 
  '''
  Generate next sequence using the previous sequence.
  
  Args:
    - H: latent representation
    - T: input time information
    
  Returns:
    - S: generated sequence based on the latent representations generated by the generator
  '''          
  with tf.compat.v1.variable_scope("supervisor", reuse = tf.compat.v1.AUTO_REUSE):
    e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers-1)])
    e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
    S = slim.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
  return S




if True:
  # Set number of samples and its dimensions
  seq_len = 24
  no, dim = 10000, 5
  ori_data = sine_data_generation(no, seq_len, dim)

parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 2
parameters['num_layer'] = 2
parameters['iterations'] = 100
parameters['batch_size'] = 32

timegan(ori_data, parameters)