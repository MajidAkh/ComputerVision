import numpy as np
import matplotlib.pyplot as plt
import time

class TwoLayerNet(object):


  """
  La couche a une dimension d'entrée de
  N, une dimension de couche cachée de dimension H, et effectue une classification sur les classes C.
  Nous formons le réseau avec une fonction de perte softmax et une régularisation L2 sur le
  matrices de poids. Le réseau utilise une non-linéarité ReLU après la première
  couche connectée.
  En d'autres termes, le réseau a l'architecture suivante:
  entrée - couche entièrement connectée - ReLU - couche entièrement connectée - softmax
  Les sorties de la deuxième couche entièrement connectée sont les scores de chaque classe.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4, 
    init_method="Normal"):

    """
    On initalise le model, les poids sont initialisé à de petite valeur aléatoire et les bias 
    à 0.
    Les poids et les biais sont stocké dans la variable self.params, qui est un dictionnaire avec
    les cléfs suivantes:
    W1: Les poids de la première couche; de dimension (D, H)
    b1: Les biais de la première couche; de dimension (H,)
    W2: Les poids de la seconde couche; de dimension (H, C)
    b2: Les biais de la seconde couche; hde dimension (C,)
    Inputs:
    - input_size: De dimension D des données d'entrées.
    - hidden_size: Le nombre de neuronnes H dans la couche caché (la couche intermédiaire).
    - output_size: Le nombre de classe C.



    """

    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)


    if init_method=="i":
      self.params['W1']=np.random.randn(input_size,hidden_size)/np.sqrt(input_size)
      self.params['W2']=np.random.randn(hidden_size,output_size)/np.sqrt(hidden_size)
    elif init_method=="io":
      self.params['W1']=np.random.randn(input_size,hidden_size)*np.sqrt(2.0/(input_size+hidden_size))
      self.params['W2']=np.random.randn(hidden_size,output_size)*np.sqrt(2.0/(hidden_size+output_size))
    elif init_method=="ReLU":
      self.params['W1']=np.random.randn(input_size,hidden_size)*np.sqrt(2.0/input_size)
      self.params['W2']=np.random.randn(hidden_size,output_size)*np.sqrt(2.0/(hidden_size+output_size))

  def loss(self, X, y=None, reg=0.0, dropout=0, dropMask=None,activation='Relu'):

    """
    Calcul la perte et les gradients pour les deux couches.
    Inputs:
    - X: forme des données d'entrée (N, D). Chaque X[i] est un echantillon d'entrainement.
    - y: Vecteur des labels d'entrainement. y[i] est le label pour X[i], et chaque y[i] est
      un entier dans l'interval 0 <= y[i] < C. 
    - reg: Force de regularisation.
    retourne:
    Si y est None, retourne une matrice de score de la forme (N, C) ou  score[i, c] est
    le score pour la classe c sur l'ebtrée X[i].
    Si y n'est pas None, la fonction retourne un tuple:
    - loss: La perte pour l'echantillon d'entrainement
    - grads: Dictionnaire mappant les noms des paramètres aux gradients de ces paramètres
      en ce qui concerne la fonction de perte; a les mêmes clés que self.params.
    """
    # On decompresse les variables du dictionnaire params

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    scores = None
    # On effectue la forward pass en calculant les scores de la classe d'entrée
    # et on stock le résultat dans la variable score, un tableau de la forme (N,C)
    
    if activation=='leaky':
      inp=X.dot(W1)+b1
      a2=np.maximum(inp,.01*inp)
    else:
      a2=np.maximum(X.dot(W1)+b1,0)

    if dropout != 0 and dropout<1:
      a2*=(np.random.randn(*a2.shape)<dropout)/dropout
    elif dropout>1:
	    W2*=dropMask['W2']/(dropout-1)
	    b2*=dropMask['b2']/(dropout-1)

    scores=a2.dot(W2)+b2 # z3
   
    
    
    if y is None:
      return scores

    
    loss = None
   
    if dropout>1:
    	print (dropMask['W2'])
    exp_scores=np.exp(scores)

    a3=exp_scores/(np.sum(exp_scores,1))[:,None] #h(x)

    loss=-np.sum(np.log(a3[range(len(a3)),y]))/len(a3)+\
      0.5*reg*(np.sum(np.power(W1,2))+np.sum(np.power(W2,2)))

    grads = {}

    # Calcul le backward pass, en calculant la dériv
    #Calcul des dérivée des poids et des biais
    #on stock les résulats dans le dictionnaire grads
    # exemple : grads[W1] devra stocker le gradient de W1

    delta_3=a3
    delta_3[range(len(a3)),y]=a3[range(len(a3)),y]-1
    delta_3/=len(a3)
    grads['W2']=a2.T.dot(delta_3)+reg*W2
    grads['b2']=np.sum(delta_3,0)


    dF=np.ones(np.shape(a2))
    if activation=='leaky':
      dF[a2<0.0]=0.01
    else:
      dF[a2==0.0]=0 #activation res a2 has been ReLUed

    delta_2=delta_3.dot(W2.T)*dF
    grads['W1']=X.T.dot(delta_2)+reg*W1
    grads['b1']=np.sum(delta_2,0)

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False,

            update="SGD", arg=.99,
            dropout=0,
            activation='ReLU'):



    """
    On entraine le réseau avec une descent de gradient stochastique.
    Inputs:
    - X: un vecteur numpy de dimension (N, D) donnant les données d'entrainement
    - y: un vecteur numpy de dimension (N,) donnant les labels d'entrainement; y[i] = c signifie que
      X[i] a comme label c, avec 0 <= c < C.
    - X_val: un vecteur numpy de dimension(N_val, D)donnant les données de validation.
    - y_val: un vecteur numpy de dimension (N_val,) donnant les label de validation.
    - learning_rate: scalaire donnant le taux d'apprentissage pour l'optimisation.
    - learning_rate_decay: scalaire donnant le facteur utilisé pour decroitre le taux d'apprentissage après chaque epoch
    - reg: sclaire donnant de la force de regularisation
    - num_iters:Nombre d'itération  à faire lors de l'optimisation..
    - batch_size:Nombre d'échantillon d'entrainement à utiliser à chaque étape.
    - verbose: boolean; si vrai imprime les progression..
    """        
    
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    #### for tracking top model
    meilleur_param = dict()
    cache_params = dict()
    mailleur_acc = 0
    cache = dict()
    dropMask = dict()
    start_time = time.time()
    ####

    for it in range(num_iters):
      X_batch = None
      y_batch = None

    #creation d'un mini-lot aléatoire de donnée et d'étiquette d'entrainement et
    # stocker dans  X_batch et y_batch


      if num_train >= batch_size:
        rand_idx = np.random.choice(num_train, batch_size)
      else:
        rand_idx = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[rand_idx]
      y_batch = y[rand_idx]

      if dropout > 1:
        for param in ['W2', 'b2']:
       	  dropMask[param] = np.random.randn(*self.params[param].shape) < (dropout-1)

      loss, grads = self.loss(X_batch, y=y_batch, reg=reg,
                              dropout=dropout, dropMask=dropMask, activation=activation)
      loss_history.append(loss)
      # utilise les gradients dans le dictionnaire grads pour mettre à jour les paramétres du reseau
      # (stocké dans le dico self.params) en utilisant la déscente de gradient stochastique


      if np.isnan(grads['W1']).any() or np.isnan(grads['W2']).any() or \
        np.isnan(grads['b1']).any() or np.isnan(grads['b2']).any():
        continue

      dx = None
      for param in self.params:
        if update == "momentum":
          if not param in cache:
            cache[param] = np.zeros(grads[param].shape)
          cache[param] = arg*cache[param]-learning_rate*grads[param]
          dx = -cache[param]
        
  

        if dropout > 1:
    	    if param == 'W2' or param == 'b2':
       	      dx *= dropMask[param]
        self.params[param] -= dx

      it += 1


      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

     
      if it % iterations_per_epoch == 0:

        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)


        learning_rate *= learning_rate_decay


        if val_acc > mailleur_acc:
          mailleur_acc = val_acc
          meilleur_param = self.params.copy()
        if verbose:

         
          print('train_acc %f, val_acc %f, time %d' %(train_acc, val_acc, (time.time()-start_time)/60.0))








    self.params=meilleur_param.copy()


    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):



    """
    On utilise les poids du réseau pour prédire les étiquettes.

    Inputs:
    - X: un vecteur numpy de dimension (N, D) donnant N D-dimension données à classifier.
    Returns:
    - y_pred: un vecteur numpy de dimension (N,) donnant des étiquettes prévues pour chacun
      pour chacun des élèments de X. Pour tout i , y_pred[i] = c signifique que X[i] est prédit
      pour avoir la classe c, avec 0 <= c < C.

    """

    y_pred = None


    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    z1 = np.dot(X,W1) + b1
    a1 = np.maximum(z1,0)
    z2 = np.dot(a1,W2) + b2
    softmaxed_scores = softmax(z2)
    y_pred = np.argmax(softmaxed_scores,axis = 1)
    return y_pred

  def accuracy(self,X,y):

    acc = (self.predict(X) == y).mean()

    return acc

  


def softmax(w):

    exponent = np.exp(w)
    distance = exponent/np.sum(exponent, axis=1, keepdims = True)
    return distance 