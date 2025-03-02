import numpy as np

class OptimizerSGDLSTM:
    def __init__(self,learning_rate=1e-5,decay=0,momentum=0) -> None:
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.momentum=momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1/(1+self.decay*self.iterations))
    def update_params(self,layer):  
            if self.momentum:    
                if not hasattr(layer,"Uf_momentums"): #if there are no momentums to the layer we initalizs them
                    layer.Uf_momentums=np.zeros_like(layer.Uf)
                    layer.Ui_momentums=np.zeros_like(layer.Ui)
                    layer.Uo_momentums=np.zeros_like(layer.Uo)
                    layer.Ug_momentums=np.zeros_like(layer.Ug)


                    layer.Wf_momentums=np.zeros_like(layer.Wf)
                    layer.Wi_momentums=np.zeros_like(layer.Wi)
                    layer.Wo_momentums=np.zeros_like(layer.Wo)
                    layer.Wg_momentums=np.zeros_like(layer.Ug)


                    layer.bf_momentums=np.zeros_like(layer.bf)
                    layer.bi_momentums=np.zeros_like(layer.bi)
                    layer.bo_momentums=np.zeros_like(layer.bo)
                    layer.bg_momentums=np.zeros_like(layer.bg)
                
                #now the momentum parts
                Uf_updates=self.momentum*layer.Uf_momentums-self.current_learning_rate*layer.dUf
                layer.Uf_momentums=Uf_updates

                Ui_updates=self.momentum*layer.Ui_momentums-self.current_learning_rate*layer.dUi
                layer.Ui_momentums=Ui_updates
                
                Uo_updates=self.momentum*layer.Uo_momentums-self.current_learning_rate*layer.dUo
                layer.Uo_momentums=Uo_updates
                
                Ug_updates=self.momentum*layer.Ug_momentums-self.current_learning_rate*layer.dUg
                layer.Ug_momentums=Ug_updates

                Wf_updates=self.momentum*layer.Wf_momentums-self.current_learning_rate*layer.dWf
                layer.Wf_momentums=Wf_updates

                Wi_updates=self.momentum*layer.Wi_momentums-self.current_learning_rate*layer.dWi
                layer.Wi_momentums=Wi_updates
                
                Wo_updates=self.momentum*layer.Wo_momentums-self.current_learning_rate*layer.dWo
                layer.Wo_momentums=Wo_updates
            
                Wg_updates=self.momentum*layer.Wg_momentums-self.current_learning_rate*layer.dWg
                layer.Wg_momentums=Wg_updates

                bf_updates=self.momentum*layer.bf_momentums-self.current_learning_rate*layer.dbf
                layer.bf_momentums=bf_updates

                bi_updates=self.momentum*layer.bi_momentums-self.current_learning_rate*layer.dbi
                layer.bi_momentums=bi_updates
                
                bo_updates=self.momentum*layer.bo_momentums-self.current_learning_rate*layer.dbo
                layer.bo_momentums=bo_updates
                
                bg_updates=self.momentum*layer.bg_momentums-self.current_learning_rate*layer.dbg
                layer.bg_momentums=bg_updates

            else:

                Uf_updates =- self.current_learning_rate*layer.Uf
                Ui_updates =- self.current_learning_rate*layer.Ui
                Uo_updates =- self.current_learning_rate*layer.Uo
                Ug_updates =- self.current_learning_rate*layer.Ug
                Wf_updates =- self.current_learning_rate*layer.Wf
                Wi_updates =- self.current_learning_rate*layer.Wi
                Wo_updates =- self.current_learning_rate*layer.Wo
                Wg_updates =- self.current_learning_rate*layer.Wg    
                bf_updates =- self.current_learning_rate*layer.bf
                bi_updates =- self.current_learning_rate*layer.bi
                bo_updates =- self.current_learning_rate*layer.bo
                bg_updates =- self.current_learning_rate*layer.bg

               
            layer.Uf += Uf_updates
            layer.Ui += Ui_updates
            layer.Uo += Uo_updates
            layer.Ug += Ug_updates
            layer.Wf += Wf_updates
            layer.Wi += Wi_updates
            layer.Wo += Wo_updates
            layer.Wg += Wg_updates   
            layer.bf += bf_updates
            layer.bi += bi_updates
            layer.bo += bo_updates
            layer.bg += bg_updates

    def post_update_params(self):
        self.iterations+=1

            