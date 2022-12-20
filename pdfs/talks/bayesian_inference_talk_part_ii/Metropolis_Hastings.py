import os
import numpy as np
import math
import matplotlib.pyplot as plt

class MetropolisHastings():

    def __init__(self,
                 initial_position,
                 generate_proposal,
                 log_proposal_density,
                 log_posterior_density,
                 seed = 0):
        np.random.seed(seed)
        self.generate_proposal = generate_proposal
        self.log_proposal_density = log_proposal_density
        self.log_posterior_density = log_posterior_density
        self.current_log_post = log_posterior_density(initial_position)
        self.chain = [initial_position]

    def MH_sample(self, N):
        ### iterate N Metropolis-Hastings steps
        for i in range(N):
            ### perform one MH step
            # sample from uniform distribution
            u = np.random.uniform(0.0, 1.0)
            # sample from the proposal distribution
            theta_star = self.generate_proposal(self.chain[-1])
            # compute log proposal and log posterior at the proposed sample
            log_post = self.log_posterior_density(theta_star)
            # compute log acceptance probability
            log_numerator = log_post + self.log_proposal_density(self.chain[-1], theta_star)
            log_denominator = self.current_log_post + self.log_proposal_density(theta_star, self.chain[-1])
            log_alpha = log_numerator - log_denominator
            # accept / reject new sample
            if (np.log(u) < log_alpha):
                self.chain.append(theta_star)
                self.current_log_post = log_post
            else:
                self.chain.append(self.chain[-1])
    
    def plot_current_status(self, dirname, x, ap, iteration = 0, theta_star = None, accepted = None):
        current = self.chain[-1]
        if accepted is None:
            proposal_colour = 'black'
        elif accepted:
            proposal_colour = 'green'
        else:
            proposal_colour = 'red'
        
        # plot the proposal step with densities
        plot = plt.subplots(1, 2, figsize=(12,8))
        plt.subplot(1, 2, 1)
        plt.plot(x, np.exp(self.log_posterior_density(x)), 'black')
        plt.plot(x, np.exp(self.log_proposal_density(x, current)), 'magenta')
        plt.plot(current, 0, marker = 'o', color = 'blue')
        plt.scatter(self.chain, [-0.05]*len(self.chain), marker = 'o', alpha = 0.25)
        plt.vlines(x=current, ymin=0, ymax=np.exp(self.log_posterior_density(current)),
                colors='blue', linestyles=':')
        if theta_star is not None:
            plt.plot(theta_star, 0, marker = 'o', color = 'orange')
            plt.vlines(x=theta_star, ymin=0, ymax=np.exp(self.log_posterior_density(theta_star)),
                       colors='orange', linestyles=':')
            plt.hlines(y=0, xmin=current, xmax=theta_star, colors=proposal_colour, linestyles=':')
        plt.xlabel(r"$\theta$")
        plt.ylabel("density")
        plt.title("Proposal step")
        
        # plot the trace plot
        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.chain)), self.chain, color = 'blue')
        if theta_star is not None:
            if accepted is None:
                plt.plot(range(len(self.chain)-1, len(self.chain)+1),
                         [self.chain[-1], theta_star], color=proposal_colour, linestyle=':')
            elif accepted:
                plt.plot(range(len(self.chain)-1, len(self.chain)+1),
                     [self.chain[-1], theta_star], color = 'green')
            else:
                plt.plot(range(len(self.chain)-1, len(self.chain)+1),
                     [self.chain[-1], theta_star], color=proposal_colour, linestyle=':')
                plt.plot(range(len(self.chain)-1, len(self.chain)+1),
                     [self.chain[-1], self.chain[-1]], color = 'green')
        plt.xlabel("iteration")
        plt.ylabel(r"$\theta$")
        plt.title(f"Trace plot. AP of moving to proposed = {ap}")
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plot[0].savefig(dirname+f"/test_{iteration}.png")
                    
    def MH_step_plot(self, dirname, x):
        # sample from uniform distribution
        u = np.random.uniform(0.0, 1.0)
        # sample from the proposal distribution
        theta_star = self.generate_proposal(self.chain[-1])
        # compute log proposal and log posterior at the proposed sample
        log_post = self.log_posterior_density(theta_star)
        # compute log acceptance probability
        log_numerator = log_post + self.log_proposal_density(self.chain[-1], theta_star)
        log_denominator = self.current_log_post + self.log_proposal_density(theta_star, self.chain[-1])
        log_alpha = log_numerator - log_denominator
        
        ##### plot current status
        self.plot_current_status(dirname = dirname,
                                 x = x,
                                 ap = round(min([1, math.exp(log_alpha)]), 5),
                                 iteration = 2*(len(self.chain)-1),
                                 theta_star = theta_star)
        
        # accept / reject new sample
        if (np.log(u) < log_alpha):
            self.plot_current_status(dirname = dirname,
                                     x = x,
                                     ap = round(min([1, math.exp(log_alpha)]), 5),
                                     theta_star = theta_star,
                                     iteration = 2*(len(self.chain)-1)+1,
                                     accepted = True)
            self.chain.append(theta_star)
            self.current_log_post = log_post
        else:
            self.plot_current_status(dirname = dirname,
                                     x = x,
                                     ap = round(min([1, math.exp(log_alpha)]), 5),
                                     theta_star = theta_star,
                                     iteration = 2*(len(self.chain)-1)+1,
                                     accepted = False)
            self.chain.append(self.chain[-1])