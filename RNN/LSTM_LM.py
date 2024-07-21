import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from RNN.util import TextVectorizer

class LSTM_LangModel(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super(LSTM_LangModel, self).__init__()
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, h0=None, c0=None):
        input_embedded = self.embed(x)
        if h0 is None and c0 is None:
            output_lstm, (h, c) = self.lstm(input_embedded)
        else:
            output_lstm, (h, c) = self.lstm(input_embedded, (h0, c0))
        output = self.dropout(output_lstm)
        output = self.dense(output)
        return output, h, c


# --------------------Main script----------------------
def main():
    # Data
    path = "data/artificial/greek/"
    with open(path+"greek_europarl_training_100k.txt", "r", encoding="utf-8") as file:
        data = []
        for line in file:
            data.append(line)

    train, val = data[:30000], data[90000:95000]
    #train, val = data[:500], data[90000:90100]

    text_vec = TextVectorizer("word")
    text_vec.build_vocab(train)

    train_dataset = text_vec.encode_dataset(train)
    val_dataset = text_vec.encode_dataset(val)

    # Configure device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model hyperparams.
    input_size = len(text_vec.vocab) + 1
    embed_size = 512
    hidden_size = 256
    output_size = len(text_vec.vocab) + 1

    model = LSTM_LangModel(input_size, embed_size, hidden_size, output_size)
    model = model.to(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # ========Main Training Loop=========

    epochs = 35
    batch_size = 64
    train_batches = math.ceil(len(train_dataset) / batch_size)
    val_batches = math.ceil(len(val_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    print("Started Training")
    best_loss = None
    for epoch in range(1, epochs+1):
        print("Epoch {}/{}".format(epoch, epochs))
        model.train()
        train_loss = 0
        for i, (sources, targets) in enumerate(train_loader):
            print("\r[{}{}] Batch {}/{}".format(math.ceil((i+1)/train_batches*20) * "=",
                                                (20 - math.ceil((i+1)/train_batches*20)) * " ", i+1, train_batches), end="")

            sources, targets = sources.to(device), targets.to(device)
            optim.zero_grad()

            # Forward pass
            output, h, c = model(sources)

            # Reshape output & targets to work with the loss function
            targets = torch.flatten(targets, end_dim=1)
            output = torch.flatten(output, end_dim=1)

            # Calculate loss
            loss = criterion(output, targets)
            train_loss += loss

            # Backward pass & update weights
            loss.backward()
            optim.step()

        # Evaluation
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for i, (sources, targets) in enumerate(val_loader):

                sources, targets = sources.to(device), targets.to(device)
                output, h, c = model(sources)

                targets = torch.flatten(targets, end_dim=1)
                output = torch.flatten(output, end_dim=1)

                loss = criterion(output, targets)
                val_loss += loss

        epoch_train_loss = train_loss / train_batches
        epoch_val_loss = val_loss / val_batches
        print("\nLoss: (train){} (eval){}".format(epoch_train_loss, epoch_val_loss))
        if not best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), "LSTM_LM_{}_{}_{}_{}.pt".format(text_vec.mode, input_size, embed_size, hidden_size))
        elif (best_loss > epoch_val_loss):
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), "LSTM_LM_{}_{}_{}_{}.pt".format(text_vec.mode, input_size, embed_size, hidden_size))

            
if __name__ == "__main__":
    main()