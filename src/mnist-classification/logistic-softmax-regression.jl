Random.seed!(0)

logistic_softmax_predict(img_vec, w) = softmax(w[:,2:end]*img_vec .+ w[:,1])
logistic_sofmax_classifier(img_vec, w) = argmax(logistic_softmax_predict(img_vec, w)) - 1;

function train_softmax_logistic(; n_epochs=20, η = 0.001, mini_batch_size=1000, verbose=true)
    #Initilize parameters
    w = randn(10, size(A)[2])

    loss_value = 0.0

    #Training loop
    for epoch in 1:n_epochs
        # prev_loss_value = loss_value
        
        #Loop over mini-batches in epoch
        start_time = time_ns()
        for batch in Iterators.partition(1:n_train, mini_batch_size)
            dw = grad_soft_loss(A'[:, batch], onehotbatch(train_labels[batch], 0:9), w)
            w = w - η*dw
        end
        end_time = time_ns()

        if verbose
            #record/display progress
            loss_value = crossentropy(logistic_softmax_predict(X', w), onehotbatch(train_labels, 0:9))
            println("Epoch = $epoch ($(round((end_time-start_time)/1e9,digits=2)) sec) Loss = $loss_value")
        end
    end
    return w
end;

grad_soft_loss(X, y, w) = (softmax(w * X) - y) * X';
