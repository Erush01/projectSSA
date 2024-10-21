
    # n_samples = 100  # Number of time series samples
    # n_timesteps = 50  # Number of time steps in each time series
    # n_features = 1  # Each time series has one feature (e.g., one sensor reading per timestep)

    # # Create time series data
    # X = np.zeros((n_samples, n_timesteps, n_features))
    # y = np.zeros(n_samples)

    # # Class 0: Sinusoidal time series
    # for i in range(n_samples // 2):
    #     X[i, :, 0] = np.sin(np.linspace(0, 3 * np.pi, n_timesteps)) + np.random.normal(0, 0.1, n_timesteps)
    #     y[i] = 0

    # # Class 1: Random walk time series
    # for i in range(n_samples // 2, n_samples):
    #     X[i, :, 0] = np.cumsum(np.random.normal(0, 1, n_timesteps))  # Random walk
    #     y[i] = 1

    # # Split into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # x_train = torch.Tensor(x_train)
    # y_train = torch.Tensor(y_train)
    
    # x_test = torch.Tensor(x_test)
    # y_test = torch.Tensor(y_test)
    
    
    # train_dataset = TensorDataset(x_train, y_train)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # val_dataset = TensorDataset(x_test, y_test)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    
    # src_pad_idx = 0
    # trg_pad_idx = 0
    # src_vocab_size = n_timesteps
    # trg_vocab_size = 1
    
    # model = Transformer(src_vocab_size,trg_vocab_size, src_pad_idx,trg_pad_idx).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # num_epochs = 10
    
    # for epoch in range(num_epochs):
    #     model.train()
    #     train_loss = 0.0
    #     train_correct = 0
    #     train_total = 0
        
    #     for inputs, labels in train_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
            
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
            
    #         train_loss += loss.item() * inputs.size(0)
    #         _, predicted = torch.max(outputs.data, 1)
    #         train_total += labels.size(0)
    #         train_correct += (predicted == labels).sum().item()
        
    #     train_loss = train_loss / len(train_loader.dataset)
    #     train_acc = train_correct / train_total
        
    #     # Validation
    #     model.eval()
    #     val_loss = 0.0
    #     val_correct = 0
    #     val_total = 0
        
    #     with torch.no_grad():
    #         for inputs, labels in val_loader:
    #             inputs, labels = inputs.to(device), labels.to(device)
                
    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)
                
    #             val_loss += loss.item() * inputs.size(0)
    #             _, predicted = torch.max(outputs.data, 1)
    #             val_total += labels.size(0)
    #             val_correct += (predicted == labels).sum().item()
        
    #     val_loss = val_loss / len(val_loader.dataset)
    #     val_acc = val_correct / val_total
        
    #     print(f'Epoch {epoch+1}/{num_epochs}:')
    #     print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    #     print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    #     print()
        
  
    