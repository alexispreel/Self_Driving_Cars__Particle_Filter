/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    
    // Setting number of particles (user input)
    //string num_particles_s = "";
    //cout << "Number of particles?" << endl;
    //cin >> num_particles_s;
    //num_particles = atoi(num_particles_s.c_str());
    num_particles = 40;
        
    // Initializing normal distributions for noise
    default_random_engine gen;
    normal_distribution<double> noise_x(x, std[0]);
	normal_distribution<double> noise_y(y, std[1]);
	normal_distribution<double> noise_theta(theta, std[2]);
    
    // Creating particles
    for (int i = 0; i < num_particles; i++) {
        // Initializing particle
        Particle p;
        
        // Assigning noisy position
        p.id = i;
        p.x = noise_x(gen);
        p.y = noise_y(gen);
        p.theta = noise_theta(gen);
        p.weight = 1.0;
        
        // Appending particle to particles
        particles.push_back(p);
        
        // Appending vector of weights
        weights.push_back(1.0);
    }
    
    // Setting initialization flag for unique execution
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
    // Calculating constants for further use
    const double v_over_yr = velocity / yaw_rate;
    const double y_times_dt = yaw_rate * delta_t;
    
    // Initializing normal distributions for noise
    default_random_engine gen;
    normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);
    
    // Iterating through particles
    for (int i = 0; i < num_particles; i++) {
        // Extracting values for easier usage
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;
        
        // Motion update: applying motion model
        if (fabs(yaw_rate) > 0.0001) {
            // If yaw rate not zero
            p_x += v_over_yr * (sin(p_theta + y_times_dt) - sin(p_theta));
            p_y += v_over_yr * (cos(p_theta) - cos(p_theta + y_times_dt));
            p_theta += y_times_dt;
        }
        else {
            // If yaw rate = zero
            p_x = p_x + velocity * delta_t * cos(p_theta);
            p_y = p_y + velocity * delta_t * sin(p_theta);
        }
        
        // Adding noise
        p_x += noise_x(gen);
        p_y += noise_y(gen);
        p_theta += noise_theta(gen);
        
        // Updating particle
        particles[i].x = p_x;
        particles[i].y = p_y;
        particles[i].theta = p_theta;
    }
    
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> landmarks, std::vector<LandmarkObs>& observations) {
	
    // Iterating through observations
    for (int i = 0; i < observations.size(); i++) {
        // Getting current observation
        LandmarkObs obsv = observations[i];
        
        // Initializing vector of distances and associated landmark's index
        vector<double> ds;
        int l_id;
        
        // Iterating through predictions
        for (int j = 0; j < landmarks.size(); j++) {
            // Getting current prediction
            LandmarkObs ldmk = landmarks[j];
            
            // Computing Euclidean distance between observation and prediction
            double d = dist(obsv.x, obsv.y, ldmk.x, ldmk.y);
            
            // Appending vector of distances
            ds.push_back(d);
        }
        
        // Assigning closest prediction to current observation
        l_id = min_element(ds.begin(),ds.end()) - ds.begin();
        observations[i].id = landmarks[l_id].id;
    }
    
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {
	
    // Calculating constants for further use
    const double denom_x = 2.0 * pow(std_landmark[0], 2);
    const double denom_y = 2.0 * pow(std_landmark[1], 2);
    const double fact_exp = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
    
    // Iterating through particles
    for (int i = 0; i < num_particles; i++) {
        // Extracting values for easier usage
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;
        
        // Initializing weight to zero
        particles[i].weight = 0.0;
        
        // Initializing vector of ldmks_range within sensor range
        vector<LandmarkObs> ldmks_range;
        
        // Keeping only landmarks within sensor range
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            
            // Extracting landmark's id and coordinates
            int l_id = map_landmarks.landmark_list[j].id_i;
            double l_x = map_landmarks.landmark_list[j].x_f;
            double l_y = map_landmarks.landmark_list[j].y_f;
            
            // If landmark within sensor range, appending vector
            if (dist(p_x, p_y, l_x, l_y) <= sensor_range) {
                ldmks_range.push_back(LandmarkObs{l_id, l_x, l_y});
            }
        }
        
        if (ldmks_range.size() > 0) {
            // Transforming observations from vehicle coordinates into map coordinates
            vector<LandmarkObs> obs_transformed;
            for (int j = 0; j < observations.size(); j++) {
                double o_x = p_x + cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y;
                double o_y = p_y + sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y;
                obs_transformed.push_back(LandmarkObs{observations[j].id, o_x, o_y});
            }
            
            // Associating observations with closest landmark
            dataAssociation(ldmks_range, obs_transformed);
            
            // Initializing associations: landmarks' id, x and y coordinates (only for visualization)
            vector<int> associations;
            vector<double> sense_x;
            vector<double> sense_y;
            associations.clear();
            sense_x.clear();
            sense_y.clear();
            
            // Initializing particle weight
            double p_weight = 1.0;
            
            // Iterating through transformed observations
            for (int j = 0; j < obs_transformed.size(); j++) {
                // Getting observation id and coordinates
                int o_id = obs_transformed[j].id;
                double o_x = obs_transformed[j].x;
                double o_y = obs_transformed[j].y;
                
                // Getting associated landmark
                double l_x;
                double l_y;
                
                // Getting associated landmark
                for (int k = 0; k < ldmks_range.size(); k++) {
                    if (ldmks_range[k].id == o_id) {
                        // 
                        l_x = ldmks_range[k].x;
                        l_y = ldmks_range[k].y;
                        break;
                    }
                }
                
                // Calculating weight using bivariate Gaussian
                p_weight *= fact_exp * exp(- (pow(o_x - l_x, 2) / denom_x + pow(o_y - l_y, 2) / denom_y));
                
                // Appending assocations (only for visualization)
                associations.push_back(o_id);
                sense_x.push_back(o_x);
                sense_y.push_back(o_y);
            }
            
            // Updating particle's weight
            particles[i].weight = p_weight;
            
            // For visualization of assocations on simulator
            particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
        }
        //cout << "   -> particles[" << i << "].weight = " << particles[i].weight << endl;
    }
}

void ParticleFilter::resample() {
    
    // Initializing vector of resampled particles
    vector<Particle> p_resampled;
    
    // Initializing maximum weight, random index and beta parameter for resampling wheel
    double w_max;
    int index;
    double beta;
    
    // Initializing normal distribution for resampling wheel
    default_random_engine gen;
    normal_distribution<double> dist_norm(0.5, 0.3);
    
    // Getting particles weights
    for (int i = 0; i < num_particles; i++) {
        weights[i] = particles[i].weight;
    }
    
    // Getting maximum weight
    w_max = *max_element(weights.begin(), weights.end());
    
    // Drawing random particle index
    index = (int)(rand() % num_particles);
    
    // Initializing beta parameter for resampling wheel
    beta = 0.0;
    
    // Iterating through particles
    for (int i = 0; i < num_particles; i++) {
        beta += 2.0 * w_max * dist_norm(gen);
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        p_resampled.push_back(particles[index]);
    }
    
    // Updating particles
    particles = p_resampled;
    
    /*
    * Those weights are mathematically wrong. 
    * But we choose to not normalize them to improve efficiency, as only relative magnitude matters here.
    */
    
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    
    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    
    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}