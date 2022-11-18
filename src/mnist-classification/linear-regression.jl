linear_classify(square_image, bets) = argmax([([1 ; vec(square_image)])'*bets[k] for k in 1:10])-1;

function train_linear()
    
    Adag = pinv(A) # pseudo-inverse of matrix A

    tfPM(x) = x ? +1 : -1
    yDat(k) = tfPM.(onehotbatch(train_labels,0:9)'[:,k+1])
    bets = [Adag*yDat(k) for k in 0:9]; #this is the trained model (a list of 10 beta coeff vectors)

    predictions = [linear_classify(test_imgs[:,:,k], bets) for k in 1:n_test]
    confusionMatrix = [sum((predictions .== i) .& (test_labels .== j)) for i in 0:9, j in 0:9]
    return sum(diag(confusionMatrix))/n_test
end;