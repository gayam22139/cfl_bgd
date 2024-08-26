# def get_range(iters=46900, tasks=10, task_id=0, beta=3):
#     if beta <= 1:
#         peak_start = int((task_id/tasks)*iters)
#         peak_end = int(((task_id + 1) / tasks)*iters)
#         start = peak_start
#         end = peak_end
#     else:
#         start = max(int(((beta*task_id - 1)*iters)/(beta*tasks)), 0)
#         peak_start = int(((beta*task_id + 1)*iters)/(beta*tasks))
#         peak_end = int(((beta * task_id + (beta - 1)) * iters) / (beta * tasks))
#         end = min(int(((beta * task_id + (beta + 1)) * iters) / (beta * tasks)), iters)

#     return start,peak_start,peak_end,end


# print("beta = "get_range(task_id=0,beta = 2))

from models import custom_cnn

model = custom_cnn()

model_params = model.parameters()

for i,params in enumerate(model_params):
    print(i,params)


print(model)

