base = read.csv('Ecommerce Customers')

base$Email = NULL
base$Address = NULL
base$Avatar = NULL

library(caTools)
set.seed(1)
divisao = sample.split(base$Yearly.Amount.Spent, SplitRatio = 0.70)
base_treinamento = subset(base, divisao == TRUE)
base_teste = subset(base, divisao == FALSE)

regressor = lm(formula = Yearly.Amount.Spent ~ ., data = base_treinamento)
summary(regressor)

previsoes = predict(regressor, newdata = base_teste[-5])
mean(abs(base_teste[['Yearly.Amount.Spent']] - previsoes))

library(miscTools)
cc = rSquared(base_teste[['Yearly.Amount.Spent']], resid = base_teste[['Yearly.Amount.Spent']] - previsoes)

# R-squared = 0.982 e mean absolute error = 7.55