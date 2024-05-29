@torch.no_grad()

    def log_gradient(self, task ):

 

        if comm.is_main_process():

            if self.iter == self.cfg.SOLVER.LOG_GRAD_ITER:

                dir_rel = 'grad_log_moe'

                if not os.path.exists(os.path.join(self.cfg.OUTPUT_DIR, dir_rel)):

                    os.makedirs(os.path.join(self.cfg.OUTPUT_DIR, dir_rel))

                for key_task in self.grad_log:

                    np.savez(

                        os.path.join(self.cfg.OUTPUT_DIR, dir_rel, f'{key_task}_grads'),

                        **self.grad_log[key_task])

 

            if getattr(self, 'grad_log', None) is None:

 

                self.grad_log = defaultdict(dict)

 

            param_names = [g['name'] for g in self.optimizer.optimizer.param_groups]

            # param_names_size = [g['params'][0].size() for g in self.optimizer.optimizer.param_groups]

            # param_names_size2 = [

            # g[0].size() for g in self.optimizer.fp16_groups

            # ]

            # assert all([ a== b for a, b in zip(param_names_size,param_names_size2 )])

 

            if not self.cfg.MOE.MOE:

                for param_name, group in zip(param_names, self.optimizer.fp16_groups):

                    names = param_name.split('.')

 

                    if len(names) > 4 and names[3] == 'ffn' and names[4] == 'dense2' and names[5]=='weight' and (int(names[2])+1)%6==0:

                        grad = group[0].grad.clone().cpu().numpy()

                        if param_name not in self.grad_log[task]:

                            self.grad_log[task][param_name] = []

 

                        self.grad_log[task][param_name].append(grad)

            else:

                current_grad_log = defaultdict(dict)

                for param_name, group in zip(param_names, self.optimizer.fp16_groups):

                    names = param_name.split('.')

                    if len(names) > 9 and names[3] == 'ffn' and names[4] == 'dense2' and names[9]=='weight' and int(names[2]) in [3, 11]:

                        grad = group[0].grad.clone()

                        newname = '.'.join(names[:5]+names[9:])

                        current_grad_log[newname][names[8]] = grad

               

                for param_name in current_grad_log:

                    # 8 is expert number

                    if param_name not in self.grad_log[task]:

                        self.grad_log[task][param_name] = []

                    self.grad_log[task][param_name].append(torch.cat([current_grad_log[param_name][str(i)].flatten() for i in range(8)]).cpu().numpy())

               

                current_grad_log = None

 