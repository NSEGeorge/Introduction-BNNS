import Accelerate

struct Network {
    private var hiddenLayer: BNNSFilter?
    private var outputLayer: BNNSFilter?
    
    mutating func createNetwork() -> Bool {
        
        let inputToHiddenWeights: [Float] = [ -65, 66, -68, 67]
        let inputToHiddenBias: [Float] = [ 32, -35]
        let hiddenToOutputWeights: [Float] = [ -112, 117]
        let hiddenToOutputBias: [Float] = [ 53 ]
        
        let activation = BNNSActivation(function: BNNSActivationFunctionSigmoid, alpha: 0, beta: 0, iscale: 0, ioffset: 0, ishift: 0, iscale_per_channel: nil, ioffset_per_channel: nil, ishift_per_channel: nil)
        
        let inputToHiddenWeightsData = BNNSLayerData(data: inputToHiddenWeights,
                                                     data_type: BNNSDataTypeFloat32,
                                                     data_scale: 0,
                                                     data_bias: 0,
                                                     data_table: nil)
        let inputToHiddenBiasData = BNNSLayerData(data: inputToHiddenBias,
                                                  data_type: BNNSDataTypeFloat32,
                                                  data_scale: 0,
                                                  data_bias: 0,
                                                  data_table: nil)
        var inputToHiddenParams = BNNSFullyConnectedLayerParameters(in_size: 2,
                                                                    out_size: 2,
                                                                    weights: inputToHiddenWeightsData,
                                                                    bias: inputToHiddenBiasData,
                                                                    activation: activation)
        var inputDesc = BNNSVectorDescriptor(size: 2,
                                             data_type: BNNSDataTypeFloat32,
                                             data_scale: 0,
                                             data_bias: 0)
        
        
        let hiddenToOutputWeightsData = BNNSLayerData(data: hiddenToOutputWeights,
                                                      data_type: BNNSDataTypeFloat32,
                                                      data_scale: 0,
                                                      data_bias: 0,
                                                      data_table: nil)
        let hiddenToOutputBiasData = BNNSLayerData(data: hiddenToOutputBias,
                                                   data_type: BNNSDataTypeFloat32,
                                                   data_scale: 0,
                                                   data_bias: 0,
                                                   data_table: nil)
        var hiddenToOutputParams = BNNSFullyConnectedLayerParameters(in_size: 2,
                                                                     out_size: 1,
                                                                     weights: hiddenToOutputWeightsData,
                                                                     bias: hiddenToOutputBiasData,
                                                                     activation: activation)
        var hiddenDesc = BNNSVectorDescriptor(size: 2,
                                              data_type: BNNSDataTypeFloat32,
                                              data_scale: 0,
                                              data_bias: 0)
        
        hiddenLayer = BNNSFilterCreateFullyConnectedLayer(&inputDesc,
                                                          &hiddenDesc,
                                                          &inputToHiddenParams,
                                                          nil)
        
        if hiddenLayer == nil {
            print("BNNSFilterCreateFullyConnectedLayer - ошибка при создании скрытого слоя")
            return false
        }
        
        var outputDesc = BNNSVectorDescriptor(size: 1,
                                              data_type: BNNSDataTypeFloat32,
                                              data_scale: 0,
                                              data_bias: 0)
        
        outputLayer = BNNSFilterCreateFullyConnectedLayer(&hiddenDesc,
                                                          &outputDesc,
                                                          &hiddenToOutputParams,
                                                          nil)
        
        if outputLayer == nil {
            print("BNNSFilterCreateFullyConnectedLayer - ошибка при создании выходного слоя")
            return false
        }
        
        return true
    }
    
    mutating func predict(a: Float, b: Float) {
        if createNetwork() {
            let input = [a, b]
            var hidden: [Float] = [0, 0]
            var output: [Float] = [0]
            
            var status = BNNSFilterApply(hiddenLayer, input, &hidden)
            if status != 0 {
                print("BNNSFilterApply - ошибка в скрытом слое")
            }
            
            status = BNNSFilterApply(outputLayer, hidden, &output)
            if status != 0 {
                print("BNNSFilterApply - ошибка в выходном слое")
            }
            
            print("\(a) ^ \(b) = \(output[0])")
        }
    }
}

var net = Network()
net.predict(a: 0, b: 0)
net.predict(a: 0, b: 1)
net.predict(a: 1, b: 0)
net.predict(a: 1, b: 1)

