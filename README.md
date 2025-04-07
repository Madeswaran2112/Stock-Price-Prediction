# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
This project aims to develop a Recurrent Neural Network (RNN)-based model, specifically leveraging Long Short-Term Memory (LSTM) networks, to predict future stock prices based on historical data. The objective is to build a deep learning model that can learn from past stock price trends and forecast future movements with improved accuracy.



## Step 1. Data Collection
The first step involves gathering historical stock market data, which typically includes attributes such as opening price, closing price, highest and lowest prices of the day, trading volume, and timestamps. Data can be sourced from financial APIs like Yahoo Finance, Alpha Vantage, or Quandl.

## Step 2. Data Preprocessing
Raw financial data often contains noise, missing values, or inconsistent formats. Preprocessing includes cleaning the dataset, handling null values, normalizing or scaling numerical data, and transforming time series data into a suitable format for supervised learning (e.g., creating input-output sequences for models like LSTM).

## Step 3. Train-Test Split
To evaluate the performance of the model, the dataset is divided into training and testing subsets. Unlike random splits used in other machine learning tasks, time series data must be split chronologically to preserve temporal dependencies.

## Step 4. Model Building
A suitable deep learning architecture is selected based on the nature of the data. Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks, are commonly used due to their ability to capture long-term temporal dependencies in sequential data. The model is compiled with appropriate loss functions (e.g., Mean Squared Error) and optimization algorithms (e.g., Adam).

## Step 5. Model Training
The model is trained using the training dataset, allowing it to learn patterns in historical stock price movements. Techniques such as early stopping and learning rate scheduling can be employed to improve generalization and prevent overfitting.

## Step 6. Evaluation and Prediction
The trained model is evaluated on the test dataset using performance metrics such as Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and R-squared score. Once validated, the model can be used to make future predictions, which can be visualized against actual stock prices for better interpretation.



## Program
#### Name:MADESWARAN M
#### Register Number:212223040106
Include your code here
```Python 
##Define RNN Model
class RNNModel(nn.Module):
    # write your code here
    def __init__(self,input_size=1,hidden_size=64,num_layer=2,output_size=1):
      super(RNNModel,self).__init__()
      self.rnn=nn.RNN(input_size,hidden_size,num_layer,batch_first=True)
      self.fc=nn.Linear(hidden_size,output_size)
    def forward(self,x):
      out,_=self.rnn(x)
      out = self.fc(out[:,-1,:])
      return out
   





model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the Model
epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")




```
## Output

## Training Loss Over Epoches
![Screenshot 2025-04-07 155633](https://github.com/user-attachments/assets/ec5466d5-5417-4ef2-9a62-09cfce06b093)


### True Stock Price, Predicted Stock Price vs time

![Screenshot 2025-04-07 155649](https://github.com/user-attachments/assets/0ca819da-fa2b-42c4-9851-fbb189810521)


### Predictions 
![Screenshot 2025-04-07 155655](https://github.com/user-attachments/assets/0cddd253-18d0-4a0d-a93e-6ba76d64bae7)

## Result
Thus, the model was successfully trained and is capable of accurately predicting stock prices based on historical data.

