#ifndef __HANDLER_H__
#define __HANDLER_H__

#include "thread.h"
#include "wqueue.h"
#include "stdio.h"


class Worker : public Thread
{
    wqueue<WorkItem*>& m_queue;
 
  public:
    Worker(wqueue<WorkItem*>& queue) : m_queue(queue) {}
 
    void* run() {
        // Remove 1 item at a time and process it. Blocks if no items are 
        // available to process.
        for (int i = 0;; i++) 
		{
            printf("thread %lu, loop %d - waiting for item...\n",  (long unsigned int)self(), i);
            WorkItem* item = m_queue.remove();
            printf("thread %lu, loop %d - got one item\n",  (long unsigned int)self(), i);
                   
            delete item;
        }
        return NULL;
    }
};

#endif // __HANDLER_H__