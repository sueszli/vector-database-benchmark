#ifndef _CLOSEST_H_
#define _CLOSEST_H_
#include "PrecompiledHeader.h"
#include "CardSelector.h"

template<typename T, typename Target>
static inline Target* closest(vector<Target*>& cards, Limitor* limitor, Target* ref)
{
    Target* card = ref;
    float curdist = 1000000.0f; // This is bigger than any possible distance
    for (typename vector<Target*>::iterator it = cards.begin(); it != cards.end(); ++it)
    {
        if (!T::test(ref, (*it)))
            continue;
        if ((*it)->actA < 32)
            continue;
        if ((NULL != limitor) && (!limitor->select(*it)))
            continue;
        if (ref)
        {
            float dist = ((*it)->x - ref->x) * ((*it)->x - ref->x) + ((*it)->y - ref->y) * ((*it)->y - ref->y);
            if (dist < curdist)
            {
                curdist = dist;
                card = *it;
            }
        }
        else
            card = *it;
    }
    return card;
}

template<typename T, typename Target>
static inline Target* closest(vector<Target*>& cards, Limitor* limitor, float x, float y)
{
    Target* card = NULL;
    float curdist = 1000000.0f; // This is bigger than any possible distance
    for (typename vector<Target*>::iterator it = cards.begin(); it != cards.end(); ++it)
    {
        if ((*it)->actA < 32)
            continue;
        if ((NULL != limitor) && (!limitor->select(*it)))
            continue;
        float dist = ((*it)->x - x) * ((*it)->x - x) + ((*it)->y - y) * ((*it)->y - y);
        if (dist < curdist)
        {
            curdist = dist;
            card = *it;
        }
    }
    return card;
}

#endif
