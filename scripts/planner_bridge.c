/* Execute the actual Generator planner with an explicitly synthetic bridge.
 * This checks storage/lifecycle contracts, not NumPy's backend arithmetic/FPE. */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "execution.h"
#include "descriptions.h"
#define WIDTH 8
typedef struct {
    const CjExecutionPlan *plan;
    double *data[CJ_MAX_SLOTS];
    unsigned char owned[CJ_MAX_SLOTS];
    double inputs[CJ_MAX_INPUTS][WIDTH];
    int allocations, frees, transfers, calls, releases, publications;
    int live, peak, private_live, private_peak;
    int fail_stage, fail_status;
} Context;

static CjExecStatus invoke(void *opaque, uint8_t i, uint16_t binding) {
    Context *c=opaque; const CjStageDescriptor *s=&c->plan->stages[i];
    c->calls++;
    if (i==c->fail_stage) return (CjExecStatus)c->fail_status;
    uint8_t out=s->output, reuse=c->plan->reuse_source[out];
    if (reuse!=CJ_NONE) {
        assert(c->owned[reuse]);
        c->data[out]=c->data[reuse]; c->owned[out]=1; c->owned[reuse]=0;
        c->transfers++;
    } else {
        c->data[out]=calloc(WIDTH,sizeof(double)); assert(c->data[out]);
        c->owned[out]=1;c->allocations++;c->live++;
        if (!c->plan->output_storage[out]) c->private_live++;
        if(c->live>c->peak)c->peak=c->live;
        if(c->private_live>c->private_peak)c->private_peak=c->private_live;
    }
    for(int k=0;k<WIDTH;k++) {
        double v=binding+1;
        for(uint8_t role=0;role<s->input_count;role++)v+=c->data[s->inputs[role]][k];
        c->data[out][k]=v;
    }
    return CJ_EXEC_OK;
}
static void release(void *opaque,uint32_t mask) {
    Context *c=opaque;c->releases++;
    for(uint8_t v=0;v<CJ_MAX_SLOTS;v++)if(mask&(UINT32_C(1)<<v)) {
        if(c->owned[v]) {
            free(c->data[v]);c->owned[v]=0;c->frees++;c->live--;
            if(!c->plan->output_storage[v])c->private_live--;
        }
        c->data[v]=NULL;
    }
}
static void cleanup(Context *c) {
    for(uint8_t v=0;v<CJ_MAX_SLOTS;v++)if(c->owned[v]) {
        free(c->data[v]);c->owned[v]=0;c->frees++;c->live--;
    }
    assert(c->live==0 && c->frees==c->allocations);
}
static int run(const CjExecutionDescription *d,const CjExecutionPlan *p,
               int fail_stage,int fail_status,double *answer,Context *c) {
    memset(c,0,sizeof(*c));c->plan=p;c->fail_stage=fail_stage;c->fail_status=fail_status;
    for(uint8_t v=0;v<d->input_count;v++) {
        c->data[v]=c->inputs[v];
        for(int k=0;k<WIDTH;k++)c->inputs[v][k]=(v+1)*10+k;
    }
    const CjExecutionBridge bridge={invoke,invoke,release};
    CjExecStatus status=cj_execute_plan(p,&bridge,c);
    if(status==CJ_EXEC_OK) {
        assert(c->owned[p->output_value]);c->publications++;
        memcpy(answer,c->data[p->output_value],WIDTH*sizeof(double));
        for(uint8_t v=0;v<d->input_count;v++)assert(c->data[p->output_value]!=c->inputs[v]);
    } else assert(c->publications==0);
    for(uint8_t v=0;v<d->input_count;v++)for(int k=0;k<WIDTH;k++)assert(c->inputs[v][k]==(v+1)*10+k);
    cleanup(c);
    return status;
}
static void array(const char *name,const uint8_t *v,int n) {
    printf("\"%s\":[",name);for(int i=0;i<n;i++)printf("%s%u",i?",":"",v[i]);printf("]");
}
int main(void) {
    for(int i=0;i<DESCRIPTION_COUNT;i++) {
        CjExecutionDescription d=descriptions[i];
        double expected[WIDTH]={0};
        for(int permissions=0;permissions<=1;permissions++) {
            CjExecutionDescription variant=d;
            if(!permissions)for(uint8_t s=0;s<d.stage_count;s++)variant.stages[s].reuse_candidates=0;
            CjExecutionPlan p;assert(cj_synthesize_plan(&variant,&p));
            Context c;double answer[WIDTH];assert(run(&variant,&p,-1,0,answer,&c)==CJ_EXEC_OK);
            if(!permissions)memcpy(expected,answer,sizeof(answer));
            else assert(memcmp(expected,answer,sizeof(answer))==0);
            int failures=0;
            for(uint8_t s=0;s<d.stage_count;s++)for(int status=-1;status<=0;status++) {
                Context failure;double unused[WIDTH];
                assert(run(&variant,&p,s,status,unused,&failure)==status);failures++;
            }
            printf("{\"description\":%d,\"permissions\":%d,\"allocations\":%d,\"allocation_bytes\":%zu,\"peak_private_bytes\":%zu,\"peak_total_bytes\":%zu,\"physical_frees\":%d,\"owner_transfers\":%d,\"stage_calls\":%d,\"release_calls\":%d,\"publications\":%d,\"failure_checks\":%d,\"plan\":{",i,permissions,c.allocations,c.allocations*WIDTH*sizeof(double),c.private_peak*WIDTH*sizeof(double),c.peak*WIDTH*sizeof(double),c.frees,c.transfers,c.calls,c.releases,c.publications,failures);
            array("order",p.order,d.stage_count);printf(",");
            array("producer",p.producer,d.value_count);printf(",");
            array("last_use",p.last_use,d.value_count);printf(",");
            array("storage_slot",p.storage_slot,d.value_count);printf(",");
            array("reuse_source",p.reuse_source,d.value_count);printf(",");
            array("output_storage",p.output_storage,d.value_count);printf(",\"release_mask\":[");
            for(uint8_t s=0;s<d.stage_count;s++)printf("%s%u",s?",":"",p.release_mask[s]);
            printf("]},\"correct\":true}\n");
        }
        /* Lower/prepare rejects a cycle before any bridge allocation. */
        d.stages[0].after|=1;CjExecutionPlan invalid;
        assert(!cj_synthesize_plan(&d,&invalid));
    }
    return 0;
}
