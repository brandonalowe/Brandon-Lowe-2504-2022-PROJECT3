linear_classify(square_image, bets) = argmax([([1 ; vec(square_image)])'*bets[k] for k in 1:10])-1

function LinearRegression(train_imgs, train_labels, test_imgs, test_labels)
    X = vcat([vec(train_imgs[:,:,k])' for k in 1:length(train_labels)]...)


    A = [ones(length(train_labels)) X];
    Adag = pinv(A);
    @show size(Adag);

    tfPM(x) = x ? +1 : -1
    yDat(k) = tfPM.(onehotbatch(train_labels,0:9)'[:,k+1])
    bets = [Adag*yDat(k) for k in 0:9]; #this is the trained model (a list of 10 beta coeff vectors)

    predictions = [linear_classify(test_imgs[:,:,k], bets) for k in 1:length(test_labels)]
    confusionMatrix = [sum((predictions .== i) .& (test_labels .== j)) for i in 0:9, j in 0:9]
    loss = sum(diag(confusionMatrix))/length(test_labels)

    return loss
end

function train_linear(; ;num_epochs = 20, η = 0.01)

end