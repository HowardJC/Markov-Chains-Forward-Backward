#!/usr/bin/env python3
from scipy.special import logsumexp

''' Estimate mu, theta_l, and theta_h via EM.

Arguments:
    -f: sequence to read in
    -mu, -theta_h, -theta_l: initializations for parameter values
Outputs:
    EM estimates for theta_h, theta_l, and mu
    em_<mu>_<theta_h>_<theta_l>.png - file containing plot of log likelihoods
                                      over the EM iterations
                                      (see ```saveplot```)

Example Usage:
    python 1a.py -f hmm-sequence.fa -mu 0.05 -theta_h 0.6 -theta_l 0.4
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import math
'''Reads the fasta file and outputs the sequence to analyze.
Arguments:
	filename: name of the fasta file
Returns:
	s: string with relevant sequence
'''


def read_fasta(filename):
    with open(filename, "r") as f:
        s = ""
        for l in f.readlines()[1:]:
            s += l.strip()
        return s


''' Generates the transition and emission probabilities table given the
    parameters.
Arguments:
    mu, theta_h, theta_l: parameters as described in the problem
Returns:
    transition_probabilities, emission_probabilities
        (both are dictionaries of dictionaries)
'''


def get_probabilities(mu, theta_h, theta_l):
    transition_probabilities = {
        'h': {'h': np.log(1 - mu), 'l': np.log(mu)},
        'l': {'h': np.log(mu), 'l': np.log(1 - mu)}
    }
    emission_probabilities = {
        'h': {'A': np.log(.5 * (1 - theta_h)), 'C': np.log(.5 * theta_h),
              'G': np.log(.5 * theta_h), 'T': np.log(.5 * (1 - theta_h))},
        'l': {'A': np.log(.5 * (1 - theta_l)), 'C': np.log(.5 * theta_l),
              'G': np.log(.5 * theta_l), 'T': np.log(.5 * (1 - theta_l))},
    }

    return transition_probabilities, emission_probabilities


''' Outputs the forward and backward probabilities of a given observation.
Arguments:
	obs: observed sequence of emitted states (list of emissions)
	trans_probs: transition log probabilities (dictionary of dictionaries)
	emiss_probs: emission log probabilities (dictionary of dictionaries)
	init_probs: initial log probabilities for each hidden state (dictionary)
Returns:
	F: matrix of forward probabilities
        likelihood_f: P(obs) calculated using the forward algorithm
	B: matrix of backward probabilities
        likelihood_b: P(obs) calculated using the backward algorithm
	R: matrix of posterior probabilities
'''


def forward_backward(obs, trans_probs, emiss_probs, init_probs):
    ''' Complete this function. '''

    ''' Complete this function. '''

    forward_probs = []
    backward_probs = []
    I = len(init_probs)  # Number of states
    N = len(obs)  # Length of observation sequence

    A = np.array([list(trans_probs['h'].values()), list(trans_probs['l'].values())])
    C = np.array(list(init_probs.values()))

    D = np.zeros((I, N)).astype(np.longdouble)
    E = np.zeros((I, N - 1)).astype(np.longdouble)

    O = list(obs)
    for index, element in enumerate(O):
        if element == "A":
            O[index] = 0
        if element == "C":
            O[index] = 1
        if element == "G":
            O[index] = 2
        if element == "T":
            O[index] = 3

    B = np.zeros(shape=(2, 4))
    B[0] = np.array(list(emiss_probs['h'].values())).astype(np.longdouble)
    B[1] = np.array(list(emiss_probs['l'].values())).astype(np.longdouble)

    for n in range(1):
        for i in range(I):
            temp_sum = 0
            D[i, n] = C[i] + B[i, O[n]]
        forward_probs.append(D[i, n])
    # Creates Traceback

    for n in range(1, N):
        for i in range(I):
            temp_sum = None
            for k in range(I):

                if k == 0:
                    temp_sum = np.add(A[i, k], D[k, n - 1])

                    continue

                temp_sum = np.logaddexp(np.add(A[k, i], D[k, n - 1]), temp_sum)

            D[i, n] = np.add(temp_sum, B[i, O[n]])

            E[i, n - 1] = np.argmax(temp_sum)
        forward_probs.append(temp_sum)
    for k in range(I):
        if k == 0:
            temp_sum = D[k, -1]
            continue

        temp_sum = np.logaddexp(D[k, -1], temp_sum)
    # Finds states
    S_opt = np.zeros(N).astype(np.longdouble)
    S_opt[-1] = np.argmax(D[:, -1])
    for n in range(N - 2, 0, -1):
        S_opt[n] = E[int(S_opt[n + 1]), n]

    likelihood_f = temp_sum

    I = len(init_probs)
    N = len(obs)

    A = np.array([list(trans_probs['h'].values()), list(trans_probs['l'].values())])
    C = np.array(list(init_probs.values()))

    D2 = np.zeros((I, N)).astype(np.longdouble)
    E = np.zeros((I, N - 1)).astype(np.int32)

    O = list(obs)
    for index, element in enumerate(O):
        if element == "A":
            O[index] = 0
        if element == "C":
            O[index] = 1
        if element == "G":
            O[index] = 2
        if element == "T":
            O[index] = 3

    B = np.zeros(shape=(2, 4))
    B[0] = np.array(list(emiss_probs['h'].values())).astype(np.float64)
    B[1] = np.array(list(emiss_probs['l'].values())).astype(np.float64)

    # Finds Max/Final Value
    for n in range(1):
        for i in range(I):
            temp_sum = 0
            for k in range(I):
                if k == 0:
                    temp_sum = np.add(A[k, i], 0)

                    continue
                temp_sum = np.logaddexp(np.add(A[k, i], 0), temp_sum)

            D2[i, -1] = 0
        backward_probs.append(temp_sum)

    for n in range(2, N + 1):
        for i in range(I):
            temp_sum = 0
            for k in range(I):

                if k == 0:
                    temp_sum = np.add(A[k, i], D2[k, -n + 1])
                    temp_sum = np.add(temp_sum, B[k, O[-n + 1]])
                    continue
                NewTemp = np.add(A[k, i], D2[k, -n + 1])
                temp_sum = np.logaddexp(np.add(NewTemp, B[k, O[-n + 1]]), temp_sum)

            D2[i, -n] = temp_sum
        backward_probs.append(temp_sum)
    for k in range(I):
        if k == 0:
            NewTemp = np.add(C[k], D2[k, 0])
            temp_sum = np.add(NewTemp, B[k, O[0]])
            continue
        NewTemp = np.add(C[k], D2[k, 0])

        temp_sum = np.logaddexp(np.add(NewTemp, B[k, O[0]]), temp_sum)

    # Finds States
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D[:, -1])
    for n in range(N - 2, 0, -1):
        S_opt[n] = E[int(S_opt[n + 1]), n]

    likelihood_b = temp_sum

    CombinedForwardBackward = np.add(D, D2)

    P = np.logaddexp(D[0] + D2[0], D[1] + D2[1])

    # Posterior=((np.exp(D)*np.exp(D2))/np.exp(P))

    Posterior = np.subtract(np.add(D, D2), P)
    Posterior = np.exp(Posterior)

    return (D, D2, likelihood_f, likelihood_b, Posterior)


''' Performs 1 EM step.
Arguments:
    fb_values: relevant output variables from forward-backward
    obs: the sequence under analysis
    tp: transition probabilities in log space
    ep: emission probabilities in log space
Returns:
    tp: updated transition probabilities, in log space
    ep: updated emission probabilities, in log space
'''


def em(fb_output, obs, tp, ep, likelihood_f, likelihood_b, Posterior, init_probs, logs):
    O = list(obs)

    for index, element in enumerate(O):
        if element == "A":
            O[index] = 0
        if element == "C":
            O[index] = 1
        if element == "G":
            O[index] = 2
        if element == "T":
            O[index] = 3

    states = [0, 1]

    # Forward and backwards are inserted into a matrix so it can be easily added and retrieved later
    def A(forward, backward):

        Ak = np.zeros(shape=(2, len(obs) - 1, 2))

        for i in range(len(obs) - 1):
            for j in range(2):
                for k in range(2):
                    Ak[j, i, k] = forward[j, i] + backward[k, i + 1] + tp[j, k] + ep[
                        k, O[i + 1]] - likelihood_f

        return Ak

    # EK is added to a matrix in regards to the current base and position using the current likelihood
    #and forward and backwards which is changed at the end of the iteration
    def E(forward, backward):

        Ek = np.zeros(
            shape=(2, len(obs)))

        for i in range(len(obs)):
            for j in range(len(states)):
                Ek[j, i] = (forward[j, i] + backward[j, i])-likelihood_f

        return Ek

    # Iterating until an answer is found

    for iteration in range(200):

        print('Turn: ', iteration + 1)

        Ak = A(fb_output[0], fb_output[1])
        Ek = E(fb_output[0], fb_output[1])

        #Creation of matrices to collect and e
        a = np.zeros((len(states), len(states)))
        e = np.zeros((len(states), 4))

        #Collecting values for each state along each base in sequence
        for j in range(len(states)):
            for i in range(len(states)):
                for t in range(len(obs) - 1):
                    if a[j, i] == 0:
                        a[j, i] = Ak[j, t, i]
                        continue


                    a[j, i] = np.logaddexp(a[j, i], Ak[j, t, i])

                    denomenator_a = [Ak[j, t_x, i_x]  for t_x in range(len(obs) - 1) for i_x in
                                     range(2) ]



                    denomenator_a = logsumexp(denomenator_a)


        #Adjusting the matrix as there is only two actual states, G/C and not G/C
        a[0, 0] = np.logaddexp(a[0, 0], a[1, 1]) - denomenator_a
        a[1, 1] = a[0, 0]
        a[1, 0] = np.logaddexp(a[1, 0], a[0, 1]) - denomenator_a
        a[0, 1] = a[1, 0]



        #Collecting values the same as in a
        for j in range(2):  # states
            for i in [1, 2]:  # seq
                indices = [idx for idx, val in enumerate(O) if (val == 1 or val == 2)]
                numerator_b = logsumexp(Ek[j, indices])
                denom_e = logsumexp(Ek[j, :])

                e[j, i] = numerator_b - denom_e
        #Adjusting the matrix again as there is only one mu value
        e[0, 0] = np.log(1-np.exp(e[0,1]))
        e[0, 3] = e[0, 0]
        e[1, 0] = np.log(1-np.exp(e[1,1]))
        e[1, 3] = e[1, 0]

        #Changes the emiss and trans probs so they can be used again in forward/backward
        emiss_probs = {'h': {'A': e[0][0], 'C': e[0][1], 'G': e[0][2], 'T': e[0][3]},
                       'l': {'A': e[1][0], 'C':e[1][1], 'G':e[1][2], 'T': e[1][3]}}
        trans_probs = {'h': {'h': a[0][0], 'l': a[0][1]}, 'l': {'h': a[1][0], 'l': a[1][1]}}


        #Checks ffor the difference between the new and old likelihood
        _, _, newlikelihood, _, _ = forward_backward(obs, trans_probs, emiss_probs, init_probs)

        diff = math.fabs(likelihood_f - newlikelihood)

        #Difference in log
        print("Likelihood: ",likelihood_f)
        print('Difference: ', diff)

        #Appends the likelihoods to the logs
        logs.append(likelihood_f)

        if (diff < 0.00001):
            break

        forward, backward, likelihood_b, likelihood_f, Posterior = forward_backward(obs, trans_probs, emiss_probs, init_probs)

        fb_output = (forward, backward)

        #Copies the matrix into tp and ep for easier use with defined A and E functions
        tp = a.copy()
        ep = e.copy()



#Note, Posteriors isnt actually needed for the function. It is only there as a reference.
    return (emiss_probs, trans_probs)


''' Helper function to save plot of log likelihoods over iterations to file for
    visualization.
Arguments:
    log_likelihoods: list of log likelihoods over iterations
    init_mu, init_theta_h, init_theta_l: the initial values of parameters used
        (for naming the file containing the plot)
Outputs:
    plot of log likelihoods to file
'''


def saveplot(log_likelihoods, mu, theta_h, theta_l):
    plt.title("EM log likelihoods with initialization %.2f, %.2f, %.2f" % (mu, theta_h, theta_l))
    plt.xlabel("Iteration")
    plt.ylabel("Log likelihood")
    plt.plot(range(len(log_likelihoods)), log_likelihoods, 'r-')
    plt.savefig("em_%.2f_%.2f_%.2f.png" % (mu, theta_h, theta_l))


''' Uses EM to infer the parameters ```mu, theta_h, theta_l```, iterating until
    a valid stopping condition is reached.
Arguments:
    sequence: sequence data to train on
    mu: the value of mu to use for initializing the transition probabilities
    theta_h, theta_l: parameters of the emission probability distributions
Returns:
    mu: parameter of trained transition probability distribution
    theta_h, theta_l: parameters of trained emission probability distribution
'''


def train(sequence, mu, theta_h, theta_l, delta=1e-4):
    init_mu, init_theta_h, init_theta_l = mu, theta_h, theta_l
    trans_probs, emiss_probs = get_probabilities(mu, theta_h, theta_l)
    init_probs = {'h': np.log(0.5), 'l': np.log(0.5)}
    log_likelihoods = []  # list of log likelihoods from each iteration
    pass  # ''' Your EM code goes here. '''
    trans = np.array([list(trans_probs['h'].values()), list(trans_probs['l'].values())])

    #Adds emiss to a starter array
    emiss = np.zeros(shape=(2, 4))
    emiss[0] = np.array(list(emiss_probs['h'].values())).astype(np.longdouble)
    emiss[1] = np.array(list(emiss_probs['l'].values())).astype(np.longdouble)

    #Check to see forward and backward actually works and outputs
    forward, backward, likelihood_f, likelihood_b, Posterior = forward_backward(sequence, trans_probs, emiss_probs,
                                                                                init_probs)
    #Actually goes into and starts the iteration.
    trans, emiss = em((forward, backward), sequence, trans, emiss, likelihood_f, likelihood_b, Posterior, init_probs,
                      log_likelihoods)

    saveplot(log_likelihoods, init_mu, init_theta_h, init_theta_l)


    theta_l = np.exp(trans['l']['G']) / (np.exp(trans['l']['G']) + np.exp(trans['l']['A']))
    theta_h = np.exp(trans['h']['G']) / (np.exp(trans['h']['G']) + np.exp(trans['h']['A']))

    mu = np.exp(emiss['h']['l']) / (np.exp(emiss['h']['l']) + np.exp(emiss['h']['h']))
    return mu, theta_h, theta_l


def main():
    parser = argparse.ArgumentParser(description='Compute mu, theta_l, and theta_h via EM.')
    parser.add_argument('-f', action="store", dest="f", type=str, default='hmm-sequence.fa')
    parser.add_argument('-mu', action="store", dest="mu", type=float, default=0.05)
    parser.add_argument('-theta_h', action="store", dest="theta_h", type=float, default=0.6)
    parser.add_argument('-theta_l', action="store", dest="theta_l", type=float, default=0.4)

    args = parser.parse_args()
    sequence = read_fasta(args.f)
    mu = args.mu
    theta_h = args.theta_h
    theta_l = args.theta_l

    mu, theta_h, theta_l = train(sequence, mu, theta_h, theta_l)
    print("theta_h: %.5f\ntheta_l: %.5f\nmu: %.5f" % (theta_h, theta_l, mu))


if __name__ == '__main__':
    main()
