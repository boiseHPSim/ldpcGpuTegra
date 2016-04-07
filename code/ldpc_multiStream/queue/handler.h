#ifndef __HANDLER_H__
#define __HANDLER_H__

#include "thread.h"
#include "wqueue.h"
#include "stdio.h"
#include  <cstring>
#include "./timer/CTimer.h"
#include "./trame/CTrame.h"
#include "./awgn_channel/CChanel_AWGN_SIMD.h"
#include "./ber_analyzer/CErrorAnalyzer.h"
#include "./terminal/CTerminal.h"

#include "./decoder_ms/CGPU_Decoder_MS_SIMD.h"
#include "./decoder_oms/CGPU_Decoder_OMS_SIMD.h"
#include "./decoder_nms/CGPU_Decoder_NMS_SIMD.h"
#include "./decoder_2nms/CGPU_Decoder_2NMS_SIMD.h"
#include "./decoder_oms_v2/CGPU_Decoder_MS_SIMD_v2.h"

#include "./matrix/constantes_gpu.h"

class WorkItem
{
	CTrame* simData;
	CChanel_AWGN_SIMD* noise;
	
  public:
    WorkItem(CTrame * data, CChanel_AWGN_SIMD* noise);
    ~WorkItem();
	CTrame* getData(){return simData;}
	CChanel_AWGN_SIMD* getNoise(){return noise;}
};

class Worker : public Thread
{
    wqueue<WorkItem*>& m_queue;
	CGPUDecoder* decoder;
	int frameErrorLimit;
	int numberIter;
	int numberThreadOnGpu;
 
  public:
    Worker(wqueue<WorkItem*>& queue, const char* DecoderType, int numThreadOnGpu, int FERL, int numberIter);
 
    void* run();
};

#endif // __HANDLER_H__