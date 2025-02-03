                Auto-Encoding   Variational Bayes

               Diederik P. Kingma        Max Welling
              Machine Learning Group  Machine Learning Group
              Universiteit van Amsterdam Universiteit van Amsterdam
              dpkingma@gmail.com     welling.max@gmail.com

                          Abstract

            How can we perform efficient inference and learning in directed probabilistic
            models, in the presence of continuous latent variables with intractable posterior
            distributions, and large datasets? We introduce a stochastic variational inference
            and learning algorithm that scales to large datasets and, under some mild differ-
            entiability conditions, even works in the intractable case. Our contributions are
            two-fold. First, we show that a reparameterization of the variational lower bound
            yields a lower bound estimator that can be straightforwardly optimized using stan-
            dard stochastic gradient methods. Second, we show that for i.i.d. datasets with
            continuous latent variables per datapoint, posterior inference can be made espe-
            cially efficient by fitting an approximate inference model (also called a recogni-
            tion model) to the intractable posterior using the proposed lower bound estimator.
            Theoretical advantages are reflected in experimental results.

         1 Introduction

         How can we perform efficient approximate inference and learning with directed probabilistic models
         whose continuous latent variables and/or parameters have intractable posterior distributions? The
         variational Bayesian (VB) approach involves the optimization of an approximation to the intractable
         posterior. Unfortunately, the common mean-field approach requires analytical solutions of expecta-
         tions w.r.t. the approximate posterior, which are also intractable in the general case. We show how a
         reparameterization of the variational lower bound yields a simple differentiable unbiased estimator
         of the lower bound; this SGVB (Stochastic Gradient Variational Bayes) estimator can be used for ef-
         ficient approximate posterior inference in almost any model with continuous latent variables and/or
         parameters, and is straightforward to optimize using standard stochastic gradient ascent techniques.

         For the case of an i.i.d. dataset and continuous latent variables per datapoint, we propose the Auto-
         Encoding VB (AEVB) algorithm. In the AEVB algorithm we make inference and learning especially
         efficient by using the SGVB estimator to optimize a recognition model that allows us to perform very
         efficient approximate posterior inference using simple ancestral sampling, which in turn allows us
         to efficiently learn the model parameters, without the need of expensive iterative inference schemes
         (such as MCMC) per datapoint. The learned approximate posterior inference model can also be used
         for a host of tasks such as recognition, denoising, representation and visualization purposes. When
         a neural network is used for the recognition model, we arrive at the variational auto-encoder.

         2 Method

         The strategy in this section can be used to derive a lower bound estimator (a stochastic objective
         function) for a variety of directed graphical models with continuous latent variables. We will restrict
         ourselves here to the common case where we have an i.i.d. dataset with latent variables per datapoint,
         and where we like to perform maximum likelihood (ML) or maximum a posteriori (MAP) inference
         on the (global) parameters, and variational inference on the latent variables. It is, for example,

                            1
2202

ceD

01

## ]LM.tats[

# 11v4116.2131:viXra



![image 0](0)



![image 1](1)



![image 2](2)

                        φ     z     θ

                            x

                             N

         Figure 1: The type of directed graphical model under consideration. Solid lines denote the generative
         model p (z)p (x|z), dashed lines denote the variational approximation q (z|x) to the intractable
            θ   θ                                φ
         posterior p (z|x). The variational parameters φ are learned jointly with the generative model pa-
             θ
         rameters θ.

         straightforward to extend this scenario to the case where we also perform variational inference on
         the global parameters; that algorithm is put in the appendix, but experiments with that case are left to
         future work. Note that our method can be applied to online, non-stationary settings, e.g. streaming
         data, but here we assume a fixed dataset for simplicity.

         2.1 Problem scenario

         Let us consider some dataset X = {x(i)}N consisting of N i.i.d. samples of some continuous
                          i=1
         or discrete variable x. We assume that the data are generated by some random process, involving
         an unobserved continuous random variable z. The process consists of two steps: (1) a value z(i)
         is generated from some prior distribution p θ∗ (z); (2) a value x(i) is generated from some condi-
         tional distribution p θ∗ (x|z). We assume that the prior p θ∗ (z) and likelihood p θ∗ (x|z) come from
         parametric families of distributions p (z) and p (x|z), and that their PDFs are differentiable almost
                       θ     θ
         everywhere w.r.t. both θ and z. Unfortunately, a lot of this process is hidden from our view: the true
         parameters θ∗ as well as the values of the latent variables z(i) are unknown to us.

         Very importantly, we do not make the common simplifying assumptions about the marginal or pos-
         terior probabilities. Conversely, we are here interested in a general algorithm that even works effi-
         ciently in the case of:

           1. Intractability: the case where the integral of the marginal likelihood p (x) =
                                             θ
            (cid:82)
             p (z)p (x|z) dz is intractable (so we cannot evaluate or differentiate the marginal like-
              θ  θ
            lihood), where the true posterior density p (z|x) = p (x|z)p (z)/p (x) is intractable
                              θ      θ   θ   θ
            (so the EM algorithm cannot be used), and where the required integrals for any reason-
            able mean-field VB algorithm are also intractable. These intractabilities are quite common
            and appear in cases of moderately complicated likelihood functions p (x|z), e.g. a neural
                                        θ
            network with a nonlinear hidden layer.

           2. A large dataset: we have so much data that batch optimization is too costly; we would like
            to make parameter updates using small minibatches or even single datapoints. Sampling-
            based solutions, e.g. Monte Carlo EM, would in general be too slow, since it involves a
            typically expensive sampling loop per datapoint.

         We are interested in, and propose a solution to, three related problems in the above scenario:

           1. Efficient approximate ML or MAP estimation for the parameters θ. The parameters can be
            of interest themselves, e.g. if we are analyzing some natural process. They also allow us to
            mimic the hidden random process and generate artificial data that resembles the real data.

           2. Efficient approximate posterior inference of the latent variable z given an observed value x
            for a choice of parameters θ. This is useful for coding or data representation tasks.

           3. Efficient approximate marginal inference of the variable x. This allows us to perform all
            kinds of inference tasks where a prior over x is required. Common applications in computer
            vision include image denoising, inpainting and super-resolution.

                            2



![image 3](3)

         For the purpose of solving the above problems, let us introduce a recognition model q (z|x): an
                                            φ
         approximation to the intractable true posterior p (z|x). Note that in contrast with the approximate
                            θ
         posterior in mean-field variational inference, it is not necessarily factorial and its parameters φ are
         not computed from some closed-form expectation. Instead, we’ll introduce a method for learning
         the recognition model parameters φ jointly with the generative model parameters θ.

         From a coding theory perspective, the unobserved variables z have an interpretation as a latent
         representation or code. In this paper we will therefore also refer to the recognition model q (z|x)
                                             φ
         as a probabilistic encoder, since given a datapoint x it produces a distribution (e.g. a Gaussian)
         over the possible values of the code z from which the datapoint x could have been generated. In a
         similar vein we will refer to p (x|z) as a probabilistic decoder, since given a code z it produces a
                     θ
         distribution over the possible corresponding values of x.

         2.2 The variational bound

         The marginal likelihood is composed of a sum over the marginal likelihoods of individual datapoints
         log p (x(1), · · · , x(N )) = (cid:80)N log p (x(i)), which can each be rewritten as:
           θ            i=1  θ
                log p (x(i)) = D (q (z|x(i))||p (z|x(i))) + L(θ, φ; x(i)) (1)
                  θ      KL φ      θ
         The first RHS term is the KL divergence of the approximate from the true posterior. Since this
         KL-divergence is non-negative, the second RHS term L(θ, φ; x(i)) is called the (variational) lower
         bound on the marginal likelihood of datapoint i, and can be written as:
              log p (x(i)) ≥ L(θ, φ; x(i)) = E [− log q (z|x) + log p (x, z)] (2)
                θ               qφ(z|x) φ       θ
         which can also be written as:
                                     (cid:104) (cid:105)
              L(θ, φ; x(i)) = −D (q (z|x(i))||p (z)) + E log p (x(i)|z) (3)
                      KL φ      θ    qφ(z|x(i)) θ

         We want to differentiate and optimize the lower bound L(θ, φ; x(i)) w.r.t. both the variational
         parameters φ and generative parameters θ. However, the gradient of the lower bound w.r.t. φ
         is a bit problematic. The usual (na¨ıve) Monte Carlo gradient estimator for this type of problem
         is: ∇ E [f (z)] = E (cid:2) f (z)∇ log q (z)(cid:3) (cid:39) 1 (cid:80)L f (z)∇ log q (z(l)) where
           φ qφ(z)    qφ(z)  qφ(z) φ    L l=1   qφ(z(l) ) φ
         z(l) ∼ q (z|x(i)). This gradient estimator exhibits exhibits very high variance (see e.g. [BJP12])
            φ
         and is impractical for our purposes.

         2.3 The SGVB estimator and AEVB algorithm

         In this section we introduce a practical estimator of the lower bound and its derivatives w.r.t. the
         parameters. We assume an approximate posterior in the form q (z|x), but please note that the
                                   φ
         technique can be applied to the case q (z), i.e. where we do not condition on x, as well. The fully
                        φ
         variational Bayesian method for inferring a posterior over the parameters is given in the appendix.

         Under certain mild conditions outlined in section 2.4 for a chosen approximate posterior q (z|x) we
                                            φ
         can reparameterize the random variable z ∼ q (z|x) using a differentiable transformation g ((cid:15), x)
                         (cid:101) φ                  φ
         of an (auxiliary) noise variable (cid:15):
                      z = g ((cid:15), x) with (cid:15) ∼ p((cid:15)) (4)
                      (cid:101) φ
         See section 2.4 for general strategies for chosing such an approriate distribution p((cid:15)) and function
         g ((cid:15), x). We can now form Monte Carlo estimates of expectations of some function f (z) w.r.t.
         φ
         q (z|x) as follows:
         φ

                             L
                    (cid:104) (cid:105) 1 (cid:88)
         E     [f (z)] = E f (g ((cid:15), x(i))) (cid:39) f (g ((cid:15)(l), x(i))) where (cid:15)(l) ∼ p((cid:15)) (5)
          qφ(z|x(i)) p((cid:15)) φ L    φ
                             l=1
         We apply this technique to the variational lower bound (eq. (2)), yielding our generic Stochastic
         Gradient Variational Bayes (SGVB) estimator L(cid:101)A(θ, φ; x(i)) (cid:39) L(θ, φ; x(i)):
                        L
                       1 (cid:88)
                L(cid:101)A(θ, φ; x(i)) =
                       L
                          log p
                            θ
                            (x(i), z(i,l)) − log q φ(z(i,l)|x(i))
                        l=1
                where z(i,l) = g ((cid:15)(i,l), x(i)) and (cid:15)(l) ∼ p((cid:15)) (6)
                       φ

                            3

         Algorithm 1 Minibatch version of the Auto-Encoding VB (AEVB) algorithm. Either of the two
         SGVB estimators in section 2.3 can be used. We use settings M = 100 and L = 1 in experiments.

          θ, φ ← Initialize parameters
          repeat
           XM ← Random minibatch of M datapoints (drawn from full dataset)
           (cid:15) ← Random samples from noise distribution p((cid:15))
           g ← ∇ θ,φL(cid:101)M (θ, φ; XM , (cid:15)) (Gradients of minibatch estimator (8))
           θ, φ ← Update parameters using gradients g (e.g. SGD or Adagrad [DHS10])
          until convergence of parameters (θ, φ)
          return θ, φ

         Often, the KL-divergence D (q (z|x(i))||p (z)) of eq. (3) can be integrated analytically (see
                    KL φ      θ
         appendix B), such that only the expected reconstruction error E (cid:2) log p (x(i)|z)(cid:3) requires
                                   qφ(z|x(i) ) θ
         estimation by sampling. The KL-divergence term can then be interpreted as regularizing φ, encour-
         aging the approximate posterior to be close to the prior p (z). This yields a second version of the
                                θ
         SGVB estimator L(cid:101)B (θ, φ; x(i)) (cid:39) L(θ, φ; x(i)), corresponding to eq. (3), which typically has less
         variance than the generic estimator:
                                  L
                                 1 (cid:88)
              L(cid:101)B (θ, φ; x(i)) = −D KL(q φ(z|x(i))||p
                             θ
                              (z)) +
                                 L
                                   (log p
                                      θ
                                      (x(i)|z(i,l)))
                                  l=1
              where z(i,l) = g ((cid:15)(i,l), x(i)) and (cid:15)(l) ∼ p((cid:15)) (7)
                     φ
         Given multiple datapoints from a dataset X with N datapoints, we can construct an estimator of the
         marginal likelihood lower bound of the full dataset, based on minibatches:
                                 M
                               N (cid:88)
                  L(θ, φ; X) (cid:39) L(cid:101)M (θ, φ; XM ) = L(cid:101)(θ, φ; x(i)) (8)
                               M
                                 i=1
         where the minibatch XM = {x(i)}M is a randomly drawn sample of M datapoints from the
                        i=1
         full dataset X with N datapoints. In our experiments we found that the number of samples L
         per datapoint can be set to 1 as long as the minibatch size M was large enough, e.g. M = 100.
         Derivatives ∇ θ,φL(cid:101)(θ; XM ) can be taken, and the resulting gradients can be used in conjunction
         with stochastic optimization methods such as SGD or Adagrad [DHS10]. See algorithm 1 for a
         basic approach to compute the stochastic gradients.

         A connection with auto-encoders becomes clear when looking at the objective function given at
         eq. (7). The first term is (the KL divergence of the approximate posterior from the prior) acts as a
         regularizer, while the second term is a an expected negative reconstruction error. The function g (.)
                                               φ
         is chosen such that it maps a datapoint x(i) and a random noise vector (cid:15)(l) to a sample from the
         approximate posterior for that datapoint: z(i,l) = g ((cid:15)(l), x(i)) where z(i,l) ∼ q (z|x(i)). Subse-
                             φ                φ
         quently, the sample z(i,l) is then input to function log p (x(i)|z(i,l)), which equals the probability
                               θ
         density (or mass) of datapoint x(i) under the generative model, given z(i,l). This term is a negative
         reconstruction error in auto-encoder parlance.

         2.4 The reparameterization trick

         In order to solve our problem we invoked an alternative method for generating samples from
         q (z|x). The essential parameterization trick is quite simple. Let z be a continuous random vari-
         φ
         able, and z ∼ q (z|x) be some conditional distribution. It is then often possible to express the
               φ
         random variable z as a deterministic variable z = g ((cid:15), x), where (cid:15) is an auxiliary variable with
                              φ
         independent marginal p((cid:15)), and g (.) is some vector-valued function parameterized by φ.
                      φ

         This reparameterization is useful for our case since it can be used to rewrite an expectation w.r.t
         q (z|x) such that the Monte Carlo estimate of the expectation is differentiable w.r.t. φ. A proof
         φ
                                            (cid:81)
         is as follows. Given the deterministic mapping z = g ((cid:15), x) we know that q (z|x) dz =
         p((cid:15)) (cid:81) d(cid:15) . Therefore1, (cid:82) q (z|x)f (z) dz = (cid:82) p((cid:15))f (zφ ) d(cid:15) = (cid:82) p((cid:15))f (g ((cid:15), xφ )) d(cid:15). It i folli ows
            i i         φ                         φ
          1Note that for infinitesimals we use the notational convention dz = (cid:81) dz
                                   i i

                            4



![image 4](4)

         that a differentiable estimator can be constructed: (cid:82) q (z|x)f (z) dz (cid:39) 1 (cid:80)L f (g (x, (cid:15)(l)))
                               φ          L  l=1 φ
         where (cid:15)(l) ∼ p((cid:15)). In section 2.3 we applied this trick to obtain a differentiable estimator of the
         variational lower bound.

         Take, for example, the univariate Gaussian case: let z ∼ p(z|x) = N (µ, σ2). In this case, a valid
         reparameterization is z = µ + σ(cid:15), where (cid:15) is an auxiliary noise variable (cid:15) ∼ N (0, 1). Therefore,
         E     [f (z)] = E [f (µ + σ(cid:15))] (cid:39) 1 (cid:80)L f (µ + σ(cid:15)(l)) where (cid:15)(l) ∼ N (0, 1).
         N (z;µ,σ2) N ((cid:15);0,1) L l=1

         For which q (z|x) can we choose such a differentiable transformation g (.) and auxiliary variable
             φ                                 φ
         (cid:15) ∼ p((cid:15))? Three basic approaches are:

           1. Tractable inverse CDF. In this case, let (cid:15) ∼ U(0, I), and let g ((cid:15), x) be the inverse CDF of
                                    φ
            q (z|x). Examples: Exponential, Cauchy, Logistic, Rayleigh, Pareto, Weibull, Reciprocal,
             φ
            Gompertz, Gumbel and Erlang distributions.
           2. Analogous to the Gaussian example, for any ”location-scale” family of distributions we can
            choose the standard distribution (with location = 0, scale = 1) as the auxiliary variable
            (cid:15), and let g(.) = location + scale · (cid:15). Examples: Laplace, Elliptical, Student’s t, Logistic,
            Uniform, Triangular and Gaussian distributions.
           3. Composition: It is often possible to express random variables as different transformations
            of auxiliary variables. Examples: Log-Normal (exponentiation of normally distributed
            variable), Gamma (a sum over exponentially distributed variables), Dirichlet (weighted
            sum of Gamma variates), Beta, Chi-Squared, and F distributions.

         When all three approaches fail, good approximations to the inverse CDF exist requiring computa-
         tions with time complexity comparable to the PDF (see e.g. [Dev86] for some methods).

         3 Example: Variational Auto-Encoder

         In this section we’ll give an example where we use a neural network for the probabilistic encoder
         q (z|x) (the approximation to the posterior of the generative model p (x, z)) and where the param-
         φ                                    θ
         eters φ and θ are optimized jointly with the AEVB algorithm.

         Let the prior over the latent variables be the centered isotropic multivariate Gaussian p (z) =
                                             θ
         N (z; 0, I). Note that in this case, the prior lacks parameters. We let p (x|z) be a multivariate
                                      θ
         Gaussian (in case of real-valued data) or Bernoulli (in case of binary data) whose distribution pa-
         rameters are computed from z with a MLP (a fully-connected neural network with a single hidden
         layer, see appendix C). Note the true posterior p (z|x) is in this case intractable. While there is
                             θ
         much freedom in the form q (z|x), we’ll assume the true (but intractable) posterior takes on a ap-
                    φ
         proximate Gaussian form with an approximately diagonal covariance. In this case, we can let the
         variational approximate posterior be a multivariate Gaussian with a diagonal covariance structure2:

                     log q (z|x(i)) = log N (z; µ(i), σ2(i)I) (9)
                       φ

         where the mean and s.d. of the approximate posterior, µ(i) and σ(i), are outputs of the encoding
         MLP, i.e. nonlinear functions of datapoint x(i) and the variational parameters φ (see appendix C).

         As explained in section 2.4, we sample from the posterior z(i,l) ∼ q (z|x(i)) using z(i,l) =
                                      φ
         g (x(i), (cid:15)(l)) = µ(i) + σ(i) (cid:12) (cid:15)(l) where (cid:15)(l) ∼ N (0, I). With (cid:12) we signify an element-wise
         φ
         product. In this model both p (z) (the prior) and q (z|x) are Gaussian; in this case, we can use the
                    θ           φ
         estimator of eq. (7) where the KL divergence can be computed and differentiated without estimation
         (see appendix B). The resulting estimator for this model and datapoint x(i) is:

                   J                          L
           L(θ, φ; x(i)) (cid:39) 1 (cid:88) (cid:16) 1 + log((σ(i))2) − (µ(i))2 − (σ(i))2(cid:17) + 1 (cid:88) log p (x(i)|z(i,l))
                 2         j     j     j    L     θ
                  j=1                        l=1
           where z(i,l) = µ(i) + σ(i) (cid:12) (cid:15)(l) and (cid:15)(l) ∼ N (0, I) (10)

         As explained above and in appendix C, the decoding term log p (x(i)|z(i,l)) is a Bernoulli or Gaus-
                                  θ
         sian MLP, depending on the type of data we are modelling.

          2Note that this is just a (simplifying) choice, and not a limitation of our method.

                            5

         4 Related work

         The wake-sleep algorithm [HDFN95] is, to the best of our knowledge, the only other on-line learn-
         ing method in the literature that is applicable to the same general class of continuous latent variable
         models. Like our method, the wake-sleep algorithm employs a recognition model that approximates
         the true posterior. A drawback of the wake-sleep algorithm is that it requires a concurrent optimiza-
         tion of two objective functions, which together do not correspond to optimization of (a bound of)
         the marginal likelihood. An advantage of wake-sleep is that it also applies to models with discrete
         latent variables. Wake-Sleep has the same computational complexity as AEVB per datapoint.

         Stochastic variational inference [HBWP13] has recently received increasing interest. Recently,
         [BJP12] introduced a control variate schemes to reduce the high variance of the na¨ıve gradient
         estimator discussed in section 2.1, and applied to exponential family approximations of the poste-
         rior. In [RGB13] some general methods, i.e. a control variate scheme, were introduced for reducing
         the variance of the original gradient estimator. In [SK13], a similar reparameterization as in this
         paper was used in an efficient version of a stochastic variational inference algorithm for learning the
         natural parameters of exponential-family approximating distributions.

         The AEVB algorithm exposes a connection between directed probabilistic models (trained with a
         variational objective) and auto-encoders. A connection between linear auto-encoders and a certain
         class of generative linear-Gaussian models has long been known. In [Row98] it was shown that PCA
         corresponds to the maximum-likelihood (ML) solution of a special case of the linear-Gaussian model
         with a prior p(z) = N (0, I) and a conditional distribution p(x|z) = N (x; Wz, (cid:15)I), specifically the
         case with infinitesimally small (cid:15).

         In relevant recent work on autoencoders [VLL+10] it was shown that the training criterion of un-
         regularized autoencoders corresponds to maximization of a lower bound (see the infomax princi-
         ple [Lin89]) of the mutual information between input X and latent representation Z. Maximiz-
         ing (w.r.t. parameters) of the mutual information is equivalent to maximizing the conditional en-
         tropy, which is lower bounded by the expected loglikelihood of the data under the autoencoding
         model [VLL+10], i.e. the negative reconstrution error. However, it is well known that this recon-
         struction criterion is in itself not sufficient for learning useful representations [BCV13]. Regular-
         ization techniques have been proposed to make autoencoders learn useful representations, such as
         denoising, contractive and sparse autoencoder variants [BCV13]. The SGVB objective contains a
         regularization term dictated by the variational bound (e.g. eq. (10)), lacking the usual nuisance regu-
         larization hyperparameter required to learn useful representations. Related are also encoder-decoder
         architectures such as the predictive sparse decomposition (PSD) [KRL08], from which we drew
         some inspiration. Also relevant are the recently introduced Generative Stochastic Networks [BTL13]
         where noisy auto-encoders learn the transition operator of a Markov chain that samples from the data
         distribution. In [SL10] a recognition model was employed for efficient learning with Deep Boltz-
         mann Machines. These methods are targeted at either unnormalized models (i.e. undirected models
         like Boltzmann machines) or limited to sparse coding models, in contrast to our proposed algorithm
         for learning a general class of directed probabilistic models.

         The recently proposed DARN method [GMW13], also learns a directed probabilistic model using
         an auto-encoding structure, however their method applies to binary latent variables. Even more
         recently, [RMW14] also make the connection between auto-encoders, directed proabilistic models
         and stochastic variational inference using the reparameterization trick we describe in this paper.
         Their work was developed independently of ours and provides an additional perspective on AEVB.

         5 Experiments

         We trained generative models of images from the MNIST and Frey Face datasets3 and compared
         learning algorithms in terms of the variational lower bound, and the estimated marginal likelihood.

         The generative model (encoder) and variational approximation (decoder) from section 3 were used,
         where the described encoder and decoder have an equal number of hidden units. Since the Frey
         Face data are continuous, we used a decoder with Gaussian outputs, identical to the encoder, except
         that the means were constrained to the interval (0, 1) using a sigmoidal activation function at the

          3Available at http://www.cs.nyu.edu/˜roweis/data.html

                            6

          100

          110

          120

          130

          140

          150
           105 106 107 108
          # Training samples evaluated
         L
            MNIST, N =3 MNIST, N =5 MNIST, N =10 MNIST, N =20 MNIST, N =200
               z         z         z         z         z
                 100       100       100       100

                 110       110       110       110

                 120       120       120       120

                 130       130       130       130

                 140       140       140       140

                 150       150       150       150
                  105 106 107 108 105 106 107 108 105 106 107 108 105 106 107 108

                 1600
                 1400
                 1200
                 1000
                 800
                 600
                 400
                 200
                  0
                  105 106 107 108
                 L
                   Frey Face, N =2 Frey Face, N =5 Frey Face, N =10 Frey Face, N =20
                       z 1600    z 1600    z 1600   z
                        1400      1400      1400
           Wake-Sleep (test) 1200      1200      1200
           Wake-Sleep (train) 1000     1000      1000
           AEVB (test)        800       800      800
           AEVB (train)       600       600      600
                        400       400      400
                        200       200      200
                         0         0         0
                         105 106 107 108 105 106 107 108 105 106 107 108

         Figure 2: Comparison of our AEVB method to the wake-sleep algorithm, in terms of optimizing the
         lower bound, for different dimensionality of latent space (N ). Our method converged considerably
                                z
         faster and reached a better solution in all experiments. Interestingly enough, more latent variables
         does not result in more overfitting, which is explained by the regularizing effect of the lower bound.
         Vertical axis: the estimated average variational lower bound per datapoint. The estimator variance
         was small (< 1) and omitted. Horizontal axis: amount of training points evaluated. Computa-
         tion took around 20-40 minutes per million training samples with a Intel Xeon CPU running at an
         effective 40 GFLOPS.

         decoder output. Note that with hidden units we refer to the hidden layer of the neural networks of
         the encoder and decoder.

         Parameters are updated using stochastic gradient ascent where gradients are computed by differenti-
         ating the lower bound estimator ∇ L(θ, φ; X) (see algorithm 1), plus a small weight decay term
                      θ,φ
         corresponding to a prior p(θ) = N (0, I). Optimization of this objective is equivalent to approxi-
         mate MAP estimation, where the likelihood gradient is approximated by the gradient of the lower
         bound.

         We compared performance of AEVB to the wake-sleep algorithm [HDFN95]. We employed the
         same encoder (also called recognition model) for the wake-sleep algorithm and the variational auto-
         encoder. All parameters, both variational and generative, were initialized by random sampling from
         N (0, 0.01), and were jointly stochastically optimized using the MAP criterion. Stepsizes were
         adapted with Adagrad [DHS10]; the Adagrad global stepsize parameters were chosen from {0.01,
         0.02, 0.1} based on performance on the training set in the first few iterations. Minibatches of size
         M = 100 were used, with L = 1 samples per datapoint.

         Likelihood lower bound We trained generative models (decoders) and corresponding encoders
         (a.k.a. recognition models) having 500 hidden units in case of MNIST, and 200 hidden units in case
         of the Frey Face dataset (to prevent overfitting, since it is a considerably smaller dataset). The chosen
         number of hidden units is based on prior literature on auto-encoders, and the relative performance
         of different algorithms was not very sensitive to these choices. Figure 2 shows the results when
         comparing the lower bounds. Interestingly, superfluous latent variables did not result in overfitting,
         which is explained by the regularizing nature of the variational bound.

         Marginal likelihood For very low-dimensional latent space it is possible to estimate the marginal
         likelihood of the learned generative models using an MCMC estimator. More information about the
         marginal likelihood estimator is available in the appendix. For the encoder and decoder we again
         used neural networks, this time with 100 hidden units, and 3 latent variables; for higher dimensional
         latent space the estimates became unreliable. Again, the MNIST dataset was used. The AEVB
         and Wake-Sleep methods were compared to Monte Carlo EM (MCEM) with a Hybrid Monte Carlo
         (HMC) [DKPR87] sampler; details are in the appendix. We compared the convergence speed for
         the three algorithms, for a small and large training set size. Results are in figure 3.

                            7



![image 5](5)



![image 6](6)

           100

           110

           120

           130

           140

           150

           160
            0 10 20 30 40 50 60
           # Training samples evaluated (millions)
         doohilekil-gol
         lanigraM
               N  = 1000          N  = 50000
               train       125    train

                        130

                                         Wake-Sleep (train)
                        135
                                         Wake-Sleep (test)
                        140                     MCEM (train)
                                         MCEM (test)
                        145
                                         AEVB (train)
                        150                     AEVB (test)

                        155

                        160
                         0 10 20 30 40 50 60

         Figure 3: Comparison of AEVB to the wake-sleep algorithm and Monte Carlo EM, in terms of the
         estimated marginal likelihood, for a different number of training points. Monte Carlo EM is not an
         on-line algorithm, and (unlike AEVB and the wake-sleep method) can’t be applied efficiently for
         the full MNIST dataset.

         Visualisation of high-dimensional data If we choose a low-dimensional latent space (e.g. 2D),
         we can use the learned encoders (recognition model) to project high-dimensional data to a low-
         dimensional manifold. See appendix A for visualisations of the 2D latent manifolds for the MNIST
         and Frey Face datasets.

         6 Conclusion

         We have introduced a novel estimator of the variational lower bound, Stochastic Gradient VB
         (SGVB), for efficient approximate inference with continuous latent variables. The proposed estima-
         tor can be straightforwardly differentiated and optimized using standard stochastic gradient meth-
         ods. For the case of i.i.d. datasets and continuous latent variables per datapoint we introduce an
         efficient algorithm for efficient inference and learning, Auto-Encoding VB (AEVB), that learns an
         approximate inference model using the SGVB estimator. The theoretical advantages are reflected in
         experimental results.

         7 Future work

         Since the SGVB estimator and the AEVB algorithm can be applied to almost any inference and
         learning problem with continuous latent variables, there are plenty of future directions: (i) learning
         hierarchical generative architectures with deep neural networks (e.g. convolutional networks) used
         for the encoders and decoders, trained jointly with AEVB; (ii) time-series models (i.e. dynamic
         Bayesian networks); (iii) application of SGVB to the global parameters; (iv) supervised models
         with latent variables, useful for learning complicated noise distributions.

                            8



![image 7](7)



![image 8](8)

         References

         [BCV13] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A re-
              view and new perspectives. 2013.

         [BJP12] David M Blei, Michael I Jordan, and John W Paisley. Variational Bayesian inference
              with Stochastic Search. In Proceedings of the 29th International Conference on Ma-
              chine Learning (ICML-12), pages 1367–1374, 2012.

         [BTL13] Yoshua Bengio and E´ ric Thibodeau-Laufer. Deep generative stochastic networks train-
              able by backprop. arXiv preprint arXiv:1306.1091, 2013.

         [Dev86] Luc Devroye. Sample-based non-uniform random variate generation. In Proceedings
              of the 18th conference on Winter simulation, pages 260–265. ACM, 1986.

         [DHS10] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online
              learning and stochastic optimization. Journal of Machine Learning Research, 12:2121–
              2159, 2010.

         [DKPR87] Simon Duane, Anthony D Kennedy, Brian J Pendleton, and Duncan Roweth. Hybrid
              monte carlo. Physics letters B, 195(2):216–222, 1987.

         [GMW13] Karol Gregor, Andriy Mnih, and Daan Wierstra. Deep autoregressive networks. arXiv
              preprint arXiv:1310.8499, 2013.

         [HBWP13] Matthew D Hoffman, David M Blei, Chong Wang, and John Paisley. Stochastic varia-
              tional inference. The Journal of Machine Learning Research, 14(1):1303–1347, 2013.

         [HDFN95] Geoffrey E Hinton, Peter Dayan, Brendan J Frey, and Radford M Neal. The” wake-
              sleep” algorithm for unsupervised neural networks. SCIENCE, pages 1158–1158, 1995.

         [KRL08] Koray Kavukcuoglu, Marc’Aurelio Ranzato, and Yann LeCun. Fast inference in sparse
              coding algorithms with applications to object recognition. Technical Report CBLL-
              TR-2008-12-01, Computational and Biological Learning Lab, Courant Institute, NYU,
              2008.

         [Lin89] Ralph Linsker. An application of the principle of maximum information preservation to
              linear systems. Morgan Kaufmann Publishers Inc., 1989.

         [RGB13] Rajesh Ranganath, Sean Gerrish, and David M Blei. Black Box Variational Inference.
              arXiv preprint arXiv:1401.0118, 2013.

         [RMW14] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic back-
              propagation and variational inference in deep latent gaussian models. arXiv preprint
              arXiv:1401.4082, 2014.

         [Row98] Sam Roweis. EM algorithms for PCA and SPCA. Advances in neural information
              processing systems, pages 626–632, 1998.

         [SK13] Tim Salimans and David A Knowles. Fixed-form variational posterior approximation
              through stochastic linear regression. Bayesian Analysis, 8(4), 2013.

         [SL10] Ruslan Salakhutdinov and Hugo Larochelle. Efficient learning of deep boltzmann ma-
              chines. In International Conference on Artificial Intelligence and Statistics, pages 693–
              700, 2010.

         [VLL+10] Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine
              Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep
              network with a local denoising criterion. The Journal of Machine Learning Research,
              9999:3371–3408, 2010.

         A  Visualisations

         See figures 4 and 5 for visualisations of latent space and corresponding observed space of models
         learned with SGVB.

                            9

           (a) Learned Frey Face manifold (b) Learned MNIST manifold

         Figure 4: Visualisations of learned data manifold for generative models with two-dimensional latent
         space, learned with AEVB. Since the prior of the latent space is Gaussian, linearly spaced coor-
         dinates on the unit square were transformed through the inverse CDF of the Gaussian to produce
         values of the latent variables z. For each of these values z, we plotted the corresponding generative
         p (x|z) with the learned parameters θ.
         θ



![image 11](11)



![image 12](12)



![image 13](13)



![image 14](14)

           (a) 2-D latent space (b) 5-D latent space (c) 10-D latent space (d) 20-D latent space

         Figure 5: Random samples from learned generative models of MNIST for different dimensionalities
         of latent space.

         B  Solution of −D (q (z)||p (z)), Gaussian case
                  KL  φ   θ

         The variational lower bound (the objective to be maximized) contains a KL term that can often be
         integrated analytically. Here we give the solution when both the prior p (z) = N (0, I) and the
                                      θ
         posterior approximation q (z|x(i)) are Gaussian. Let J be the dimensionality of z. Let µ and σ
                   φ
         denote the variational mean and s.d. evaluated at datapoint i, and let µ and σ simply denote the
                                     j    j
         j-th element of these vectors. Then:

                 (cid:90)     (cid:90)
                  q (z) log p(z) dz = N (z; µ, σ2) log N (z; 0, I) dz
                   θ

                                  J
                           J      1 (cid:88)
                          = − log(2π) − (µ2 + σ2)
                            2      2    j  j
                                  j=1

                            10



![image 9](9)



![image 10](10)

         And:
                 (cid:90)     (cid:90)
                  q (z) log q (z) dz = N (z; µ, σ2) log N (z; µ, σ2) dz
                  θ    θ

                                  J
                           J      1 (cid:88)
                         = − log(2π) − (1 + log σ2)
                           2      2        j
                                  j=1

         Therefore:
                          (cid:90)
               −D  ((q (z)||p (z)) = q (z) (log p (z) − log q (z)) dz
                 KL φ   θ      θ     θ      θ

                           J
                         = 1 (cid:88) (cid:0) 1 + log((σ )2) − (µ )2 − (σ )2(cid:1)
                          2         j     j   j
                           j=1

         When using a recognition model q (z|x) then µ and s.d. σ are simply functions of x and the
                       φ
         variational parameters φ, as exemplified in the text.

         C  MLP’s as probabilistic encoders and decoders

         In variational auto-encoders, neural networks are used as probabilistic encoders and decoders. There
         are many possible choices of encoders and decoders, depending on the type of data and model. In
         our example we used relatively simple neural networks, namely multi-layered perceptrons (MLPs).
         For the encoder we used a MLP with Gaussian output, while for the decoder we used MLPs with
         either Gaussian or Bernoulli outputs, depending on the type of data.

         C.1 Bernoulli MLP as decoder

         In this case let p (x|z) be a multivariate Bernoulli whose probabilities are computed from z with a
               θ
         fully-connected neural network with a single hidden layer:

                        D
                        (cid:88)
                  log p(x|z) = x log y + (1 − x ) · log(1 − y )
                          i  i     i      i
                        i=1
                   where y = f (W tanh(W z + b ) + b )   (11)
                        σ  2     1   1   2

         where f (.) is the elementwise sigmoid activation function, and where θ = {W , W , b , b } are
            σ                                       1  2 1 2
         the weights and biases of the MLP.

         C.2 Gaussian MLP as encoder or decoder

         In this case let encoder or decoder be a multivariate Gaussian with a diagonal covariance structure:
                      log p(x|z) = log N (x; µ, σ2I)
                       where µ = W h + b
                             4   4
                        log σ2 = W h + b
                             5   5
                          h = tanh(W z + b )           (12)
                               3   3

         where {W , W , W , b , b , b } are the weights and biases of the MLP and part of θ when used
             3  4  5 3 4 5
         as decoder. Note that when this network is used as an encoder q (z|x), then z and x are swapped,
                                  φ
         and the weights and biases are variational parameters φ.

         D  Marginal likelihood estimator

         We derived the following marginal likelihood estimator that produces good estimates of the marginal
         likelihood as long as the dimensionality of the sampled space is low (less then 5 dimensions), and
         sufficient samples are taken. Let p (x, z) = p (z)p (x|z) be the generative model we are sampling
                      θ     θ  θ
         from, and for a given datapoint x(i) we would like to estimate the marginal likelihood p (x(i)).
                                            θ

         The estimation process consists of three stages:

                            11

           1. Sample L values {z(l)} from the posterior using gradient-based MCMC, e.g. Hybrid Monte
            Carlo, using ∇ log p (z|x) = ∇ log p (z) + ∇ log p (x|z).
                  z   θ     z   θ    z   θ
           2. Fit a density estimator q(z) to these samples {z(l)}.

           3. Again, sample L new values from the posterior. Plug these samples, as well as the fitted
            q(z), into the following estimator:

                    (cid:32) 1 (cid:88)L q(z(l)) (cid:33)−1
                p (x(i)) (cid:39)         where z(l) ∼ p (z|x(i))
                θ      L   p (z)p (x(i)|z(l))      θ
                        θ  θ
                      l=1

         Derivation of the estimator:

                 1    (cid:82) q(z) dz (cid:82) q(z) pθ (x(i) ,z) dz
                   =      =
                            pθ (x(i) ,z)
               p (x(i)) p (x(i)) p (x(i))
                θ      θ         θ
                    (cid:90) p (x(i), z) q(z)
                   =   θ           dz
                      p (x(i)) p (x(i), z)
                      θ    θ
                    (cid:90) q(z)
                   =  p (z|x(i))  dz
                      θ    p (x(i), z)
                          θ
                    1 (cid:88)L q(z(l))
                   (cid:39)         where z(l) ∼ p (z|x(i))
                    L  p (z)p (x(i)|z(l))    θ
                       θ  θ
                     l=1

         E  Monte Carlo EM

         The Monte Carlo EM algorithm does not employ an encoder, instead it samples from the pos-
         terior of the latent variables using gradients of the posterior computed with ∇ log p (z|x) =
                                          z   θ
         ∇ log p (z) + ∇ log p (x|z). The Monte Carlo EM procedure consists of 10 HMC leapfrog
         z   θ    z   θ
         steps with an automatically tuned stepsize such that the acceptance rate was 90%, followed by 5
         weight updates steps using the acquired sample. For all algorithms the parameters were updated
         using the Adagrad stepsizes (with accompanying annealing schedule).

         The marginal likelihood was estimated with the first 1000 datapoints from the train and test sets,
         for each datapoint sampling 50 values from the posterior of the latent variables using Hybrid Monte
         Carlo with 4 leapfrog steps.

         F  Full VB

         As written in the paper, it is possible to perform variational inference on both the parameters θ and
         the latent variables z, as opposed to just the latent variables as we did in the paper. Here, we’ll derive
         our estimator for that case.

         Let p (θ) be some hyperprior for the parameters introduced above, parameterized by α. The
           α
         marginal likelihood can be written as:

                   log p (X) = D (q (θ)||p (θ|X)) + L(φ; X) (13)
                     α     KL φ    α

         where the first RHS term denotes a KL divergence of the approximate from the true posterior, and
         where L(φ; X) denotes the variational lower bound to the marginal likelihood:
                     (cid:90)
                L(φ; X) = q (θ) (log p (X) + log p (θ) − log q (θ)) dθ (14)
                      φ     θ      α      φ

         Note that this is a lower bound since the KL divergence is non-negative; the bound equals the true
         marginal when the approximate and true posteriors match exactly. The term log p (X) is composed
                                         θ
         of a sum over the marginal likelihoods of individual datapoints log p (X) = (cid:80)N log p (x(i)),
                                     θ     i=1  θ
         which can each be rewritten as:

                log p (x(i)) = D (q (z|x(i))||p (z|x(i))) + L(θ, φ; x(i)) (15)
                  θ      KL φ      θ

                            12

         where again the first RHS term is the KL divergence of the approximate from the true posterior, and
         L(θ, φ; x) is the variational lower bound of the marginal likelihood of datapoint i:
                    (cid:90) (cid:16)             (cid:17)
             L(θ, φ; x(i)) = q (z|x) log p (x(i)|z) + log p (z) − log q (z|x) dz (16)
                      φ      θ         θ     φ

         The expectations on the RHS of eqs (14) and (16) can obviously be written as a sum of three separate
         expectations, of which the second and third component can sometimes be analytically solved, e.g.
         when both p (x) and q (z|x) are Gaussian. For generality we will here assume that each of these
              θ     φ
         expectations is intractable.

         Under certain mild conditions outlined in section (see paper) for chosen approximate posteriors
         q (θ) and q (z|x) we can reparameterize conditional samples z ∼ q (z|x) as
         φ     φ                           (cid:101) φ
                      z = g ((cid:15), x) with (cid:15) ∼ p((cid:15)) (17)
                      (cid:101) φ
         where we choose a prior p((cid:15)) and a function g ((cid:15), x) such that the following holds:
                           φ
                 (cid:90) (cid:16)             (cid:17)
           L(θ, φ; x(i)) = q (z|x) log p (x(i)|z) + log p (z) − log q (z|x) dz
                   φ       θ        θ      φ

                = (cid:90) p((cid:15)) (cid:16) log p θ (x(i)|z) + log p θ (z) − log q φ(z|x)(cid:17) (cid:12) (cid:12) (cid:12) d(cid:15) (18)
                                      (cid:12)
                                       z=gφ((cid:15),x(i))

         The same can be done for the approximate posterior q (θ):
                              φ

                      θ(cid:101) = h φ(ζ) with ζ ∼ p(ζ) (19)

         where we, similarly as above, choose a prior p(ζ) and a function h (ζ) such that the following
                                     φ
         holds:
                   (cid:90)
              L(φ; X) = q (θ) (log p (X) + log p (θ) − log q (θ)) dθ
                     φ     θ      α      φ

                   (cid:90)                  (cid:12)
                                      (cid:12)
                  =  p(ζ) (log p θ (X) + log p α(θ) − log q φ(θ)) (cid:12) dζ (20)
                                      (cid:12)
                                      θ=hφ(ζ)

         For notational conciseness we introduce a shorthand notation f (x, z, θ):
                                  φ

           f (x, z, θ) = N · (log p (x|z) + log p (z) − log q (z|x)) + log p (θ) − log q (θ) (21)
           φ            θ       θ      φ        α      φ

         Using equations (20) and (18), the Monte Carlo estimate of the variational lower bound, given
         datapoint x(i), is:

                         L
                       1 (cid:88)
                  L(φ; X) (cid:39) f (x(l), g ((cid:15)(l), x(l)), h (ζ(l))) (22)
                       L   φ    φ       φ
                        l=1

         where (cid:15)(l) ∼ p((cid:15)) and ζ(l) ∼ p(ζ). The estimator only depends on samples from p((cid:15)) and p(ζ)
         which are obviously not influenced by φ, therefore the estimator can be differentiated w.r.t. φ.
         The resulting stochastic gradients can be used in conjunction with stochastic optimization methods
         such as SGD or Adagrad [DHS10]. See algorithm 1 for a basic approach to computing stochastic
         gradients.

         F.1 Example

         Let the prior over the parameters and latent variables be the centered isotropic Gaussian p (θ) =
                                             α
         N (z; 0, I) and p (z) = N (z; 0, I). Note that in this case, the prior lacks parameters. Let’s also
               θ
         assume that the true posteriors are approximatily Gaussian with an approximately diagonal covari-
         ance. In this case, we can let the variational approximate posteriors be multivariate Gaussians with
         a diagonal covariance structure:

                      log q (θ) = log N (θ; µ , σ2 I)
                        φ         θ  θ
                      log q (z|x) = log N (z; µ , σ2I)  (23)
                        φ           z  z

                            13

         Algorithm 2 Pseudocode for computing a stochastic gradient using our estimator. See text for
         meaning of the functions f , g and h .
                   φ φ   φ

         Require: φ (Current value of variational parameters)
          g ← 0
          for l is 1 to L do
           x ← Random draw from dataset X
           (cid:15) ← Random draw from prior p((cid:15))
           ζ ← Random draw from prior p(ζ)
           g ← g + 1 ∇ f (x, g ((cid:15), x), h (ζ))
               L φ φ  φ    φ
          end for
          return g

         where µ and σ are yet unspecified functions of x. Since they are Gaussian, we can parameterize
            z   z
         the variational approximate posteriors:

               q φ(θ) as θ(cid:101) = µ
                       θ
                        + σ
                         θ
                          (cid:12) ζ where ζ ∼ N (0, I)
              q (z|x) as z = µ + σ (cid:12) (cid:15) where (cid:15) ∼ N (0, I)
               φ       (cid:101) z z
         With (cid:12) we signify an element-wise product. These can be plugged into the lower bound defined
         above (eqs (21) and (22)).

         In this case it is possible to construct an alternative estimator with a lower variance, since in this
         model p (θ), p (z), q (θ) and q (z|x) are Gaussian, and therefore four terms of f can be solved
            α    θ  φ     φ                           φ
         analytically. The resulting estimator is:
                                                       
                L     J
          L(φ; X) (cid:39) L1 (cid:88) N ·  21 (cid:88) (cid:16) 1 + log((σ z(l ,) j )2) − (µ( zl ,) j )2 − (σ z(l ,) j )2(cid:17) + log p θ (x(i)z(i))
                l=1    j=1

                J
              + 1 (cid:88) (cid:16) 1 + log((σ(l) )2) − (µ(l) )2 − (σ(l) )2(cid:17) (24)
               2          θ,j   θ,j   θ,j
                j=1

         µ(i) and σ(i) simply denote the j-th element of vectors µ(i) and σ(i).
         j   j

                            14


![image 15](15)



![image 16](16)



![image 17](17)

