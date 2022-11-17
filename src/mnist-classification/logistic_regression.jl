logistic_classifier(img_vec, w, b) = logistic_predict(img_vec, w, b) .> 0.5; #Threhsold for predicting a positive sample
logistic_predict(img_vec, w, b) = sig.(w'*img_vec .+ b);

function train_logistic(;num_epochs = 20, mini_batch_size = 100, η = 0.01)
    
    #Initilize parameters
    w = randn(28*28)
    b = randn(1)

    #As a loss function for training, We'll use the binary cross entropy 
    #which takes in a probability in [0,1]
    #and an actual label in {0,1}. The probability (of Ankle boot) 
    #in [0,1] is determined by the logistic model.
    loss(x, y) = binarycrossentropy(logistic_predict(x, w, b), y);
    
    loss_value = 0.0

    #Training loop
    for epoch_num in 1:num_epochs
        
        #Loop over mini-batches in epoch
        start_time = time_ns()
        for batch in Iterators.partition(1:n_train_2_class, mini_batch_size)
            gs = gradient(()->loss(train_data_2_class'[:,batch], train_labels_2_class[batch]'), params(w, b))
            b = b - η*gs[b]
            w = w - η*gs[w]
        end
        end_time = time_ns()

        #record/display progress
        loss_value = loss(train_data_2_class', train_labels_2_class')
        println("Epoch = $epoch_num ($(round((end_time-start_time)/1e9,digits=2)) sec) Loss = $loss_value")
        
    end
    return w, b
end