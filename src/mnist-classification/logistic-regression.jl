Random.seed!(0)

σ(x) = 1/(1+float(MathConstants.e)^-x);

grad_loss(x, y, w) = (σ.(w' * x) .- y) * x';
ovo_grad_loss(x, y, w) = sum(((σ.(w' * x) .- y) * x')', dims=2);


logistic_predict(img_vec, w) = σ.(w[:,2:end]*img_vec .+ w[:,1]);
logistic_classify(square_image, w) = argmax(σ.(w*square_image))-1;
logistic_classifier(img_vec, w) = logistic_predict(img_vec, w) .> 0.5; #Threhsold for predicting a positive sample

function train_logistic(; n_epochs=20, η=0.01, mini_batch_size=1000, verbose=true)

    # Init parameters
    w = randn(10, size(A)[2])
    loss_value = 0.0

    # start mini-batching
    for epoch in 1:n_epochs
        start_time = time_ns()
        for batch in Iterators.partition(1:n_train, mini_batch_size)
            dw = grad_loss(A'[:, batch], onehotbatch(train_labels[batch], 0:9), w')
            w = w - η*dw
        end
        end_time = time_ns()

        #record/display progress
        if verbose
            loss_value = binarycrossentropy(logistic_predict(X', w), onehotbatch(train_labels, 0:9)) # using Flux.jl for simplicity
            println("Epoch = $epoch ($(round((end_time-start_time)/1e9,digits=2)) sec) Loss = $loss_value")
        end
    end
    return w
end

function ovo_train_logistic(train_data_class, train_labels_class; num_epochs = 3, mini_batch_size = 1000, η = 0.01)
    
    n_train_class = size(train_data_class)[1]

    # Init parameters
    A = [ones(n_train_class) train_data_class]
    w = randn(size(A)[2])
    loss_value = 0.0
    
    #Training loop
    for epoch_num in 1:num_epochs
        
        #Loop over mini-batches in epoch
        start_time = time_ns()
        for batch in Iterators.partition(1:n_train_class, mini_batch_size)
            dw = ovo_grad_loss(A'[:, batch], train_labels_class[batch], w)
            w = w .- η*dw
        end
        end_time = time_ns()

        #record/display progress
        loss_value = binarycrossentropy((logistic_predict(train_data_class', w'))', train_labels_class)
        println("Epoch = $epoch_num ($(round((end_time-start_time)/1e9,digits=2)) sec) Loss = $loss_value")
        
    end
    return w
end

