/*! @file PSOOptimiser.cpp
    @brief Implemenation of PSOOptimiser class
 
    @author Jason Kulk
 
  Copyright (c) 2010 Jason Kulk
 
    This file is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with NUbot.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "PSOOptimiser.h"
#include "Parameter.h"

#include "NUPlatform/NUPlatform.h"
#include "Tools/Math/StlVector.h"

#include "debug.h"
#include "nubotdataconfig.h"

/*! @brief Constructor for abstract optimiser
 	@param name the name of the optimiser. The name is used in debug logs, and is used for load/save filenames by default
 	@param parameters the initial seed for the optimisation
 */
PSOOptimiser::PSOOptimiser(std::string name, vector<Parameter> parameters) : Optimiser(name, parameters)
{
    m_c1 = 1.50;             // tune this: the literature says that these are usually set equal, and from my grid search setting them different does not have a great effect   
    m_c2 = 0.80;             // tune this:
    m_inertia = 0.60;       // tune this: this must be less than 1, and can be used to control how long it takes for the algorithm to converge (0.7 converges after about 2000)

    m_reset_limit = 10;
    m_reset_fraction = 0.05;
    m_num_particles = 30;

    m_num_dimensions = parameters.size();
    
    srand(static_cast<unsigned int> (1e6*Platform->getRealTime()*Platform->getRealTime()*Platform->getRealTime()));
    load();
    if (m_swarm_position.empty())
    	initSwarm();
    save();
}

void PSOOptimiser::initSwarm()
{
    // we want to start the swarm around initial_parameters
    debug << "Initial Swarm: " << endl;
    for (int i=0; i<m_num_particles; i++)
    {
    	m_swarm_best.push_back(m_initial_parameters);
        m_swarm_best_fitness.push_back(0);
        m_swarm_failures.push_back(0);
        
        vector<Parameter> particle = m_initial_parameters;
        vector<float> velocity = vector<float>(m_num_dimensions, 0);
        for (int j=0; j<m_num_dimensions; j++)
        {
        	float min = m_initial_parameters[j].min();
			float max = m_initial_parameters[j].max();
			particle[j].set(uniformDistribution(min, max));
			velocity[j] = normalDistribution(0, (max-min)/8);        // initial velocity between +/- range
        }
        debug << i << ": " << Parameter::getAsVector(particle) << endl;
        debug << i << ": " << velocity << endl;
        m_swarm_position.push_back(particle);
        m_swarm_velocity.push_back(velocity);
    }
    m_best = m_initial_parameters;
    m_best_fitness = 0;
}

/*! @brief Destructor for the abstract optimiser */
PSOOptimiser::~PSOOptimiser()
{
}

void PSOOptimiser::setParametersResult(float fitness)
{
    debug << "PSOOptimiser::setParametersResult fitness: " << fitness << endl;
	m_swarm_fitness.push_back(fitness);
    if (m_swarm_fitness.size() == (unsigned int) m_num_particles)
        updateSwarm();
}

vector<float> PSOOptimiser::getNextParameters()
{
    return Parameter::getAsVector(m_swarm_position[m_swarm_fitness.size()]);
}

void PSOOptimiser::updateSwarm()
{
    debug << "Fitnesses: " << m_swarm_fitness << endl;
    // update the personal and global bests
    for (int i=0; i<m_num_particles; i++)
    {
        if (m_swarm_fitness[i] > m_swarm_best_fitness[i])
        {
            m_swarm_best_fitness[i] = m_swarm_fitness[i];
            m_swarm_best[i] = m_swarm_position[i];
            m_swarm_failures[i] = 0;
        }
        else
        	m_swarm_failures[i]++;;
        
        if (m_swarm_fitness[i] > m_best_fitness)
        {
            m_best_fitness = m_swarm_fitness[i];
            m_best = m_swarm_position[i];
        }
    }
    
    debug << "Personal bests: " << m_swarm_best_fitness << endl;
    debug << "Global best: " << m_best_fitness << " " << Parameter::getAsVector(m_best) << endl;
    
    debug << "Current Swarm:" << endl;
    // update the positions and velocities of the particles
    for (int i=0; i<m_num_particles; i++)
    {
    	if (m_swarm_failures[i] < m_reset_limit)
    	{
			for (int j=0; j<m_num_dimensions; j++)
			{
				// Gaussian swarm
				float cognitivefactor = fabs(normalDistribution(0,1))*(m_swarm_best[i][j] - m_swarm_position[i][j]);
				float socialfactor = fabs(normalDistribution(0,1))*(m_best[j] - m_swarm_position[i][j]);
				m_swarm_velocity[i][j] = cognitivefactor + socialfactor;

				// PSO Swarm
				/*float cognitivefactor = c1*uniformDistribution(0,1)*(m_swarm_best[i][j] - m_swarm_position[i][j]);
				float socialfactor = c2*uniformDistribution(0,1)*(m_best[j] - m_swarm_position[i][j]);
				m_swarm_velocity[i][j] = m_inertia*m_swarm_velocity[i][j] + cognitivefactor + socialfactor;

				*/
				// I need to clip each velocity
				float max = (m_best[j].max() - m_best[j].min())/8;
				if (m_swarm_velocity[i][j] < -max)
					m_swarm_velocity[i][j] = -max;
				else if (m_swarm_velocity[i][j] > max)
					m_swarm_velocity[i][j] = max;
			}
			m_swarm_position[i] += m_swarm_velocity[i];
    	}
    	else
    	{
    		debug << "reset " << i << endl;
    		m_swarm_failures[i] = 0;
    		for (int j=0; j<m_num_dimensions; j++)
    		{
    			m_swarm_position[i][j] += normalDistribution(0, m_reset_fraction)*(m_swarm_position[i][j].max() - m_swarm_position[i][j].min());
    			m_swarm_velocity[i][j] = normalDistribution(0, (m_swarm_position[i][j].max() - m_swarm_position[i][j].min())/8);
    		}
    	}
        
        debug << "pos " << i << ": " << Parameter::getAsVector(m_swarm_position[i]) << endl;
        debug << "vel" << i << ": " << m_swarm_velocity[i] << endl;
    }
    
    // clear the fitnesses
    m_swarm_fitness.clear();
}

void PSOOptimiser::summaryTo(ostream& stream)
{
}

void PSOOptimiser::toStream(ostream& o) const
{
    o << m_c1 << " " << m_c2 << " " << m_inertia << " " << m_reset_limit << " " << m_reset_fraction << " " << m_num_particles << " " << m_num_dimensions << endl;
    
    o << m_swarm_position << endl;
    o << m_swarm_velocity << endl;
    o << m_swarm_fitness << endl;
    
    o << m_swarm_best << endl;
    o << m_swarm_best_fitness << endl;
    o << m_swarm_failures << endl;
    o << m_best << endl;
    o << m_best_fitness << endl;
}

void PSOOptimiser::fromStream(istream& i)
{
    i >> m_c1 >> m_c2 >> m_inertia >> m_reset_limit >> m_reset_fraction >> m_num_particles >> m_num_dimensions;
    
    i >> m_swarm_position;
    i >> m_swarm_velocity;
    i >> m_swarm_fitness;
    
    i >> m_swarm_best;
    i >> m_swarm_best_fitness;
    i >> m_swarm_failures;
    i >> m_best;
    i >> m_best_fitness;
}

