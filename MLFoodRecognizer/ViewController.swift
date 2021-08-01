/*
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// NOTE: A model needs to imported into the project before building and testing.

import UIKit
import Vision
import AVFoundation

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var deviceInput: AVCaptureDeviceInput!
    let session = AVCaptureSession()
    let videoQueue = DispatchQueue(label: "video",
                                   qos: .userInteractive,
                                   attributes: [],
                                   autoreleaseFrequency: .workItem)
    let videoDataOutput = AVCaptureVideoDataOutput()
    var rootLayer: CALayer! = nil
    var textOverlay = CATextLayer()
    var previewLayer: AVCaptureVideoPreviewLayer! = nil
    @IBOutlet var previewView: UIView!
    var prediction: VNClassificationObservation?
    var requests = [VNRequest]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupVideoCaptureAndLayers()
        setupVision()
        session.startRunning()
    }
    
    //Set up video caputure
    func setupVideoCaptureAndLayers() {
        // Set up the AV session
        let videoDevice = AVCaptureDevice.default(for: .video)
        do { 
            deviceInput = try AVCaptureDeviceInput(device:videoDevice!)
        } catch { return }
        session.beginConfiguration()
        session.sessionPreset = .vga640x480
        if session.canAddOutput(videoDataOutput) && session.canAddInput(deviceInput) {
            session.addInput(deviceInput)
            session.addOutput(videoDataOutput)
            videoDataOutput.setSampleBufferDelegate(self, queue: videoQueue)
        } else { return }
        session.commitConfiguration()
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        previewLayer.connection?.videoOrientation = AVCaptureVideoOrientation.landscapeRight
        rootLayer = previewView.layer
        rootLayer.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        previewLayer.frame = rootLayer.bounds
        rootLayer.addSublayer(previewLayer)
        textOverlay.alignmentMode = CATextLayerAlignmentMode.center
        textOverlay.bounds = rootLayer.bounds
        textOverlay.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        rootLayer.addSublayer(textOverlay)
    }
    
    //Set up Vision and classify images
    func setupVision() {
        guard let modelURL = Bundle.main.url(forResource: "Food101", withExtension: "mlmodelc") else { return }
        do {
            let foodVisionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            let objectRecognition = VNCoreMLRequest(model: foodVisionModel, completionHandler: { (request, error) in
                DispatchQueue.main.async(execute: {
                    if let results = request.results {
                        self.updateClassificationLabel(results)
                    }
                })
            })
            self.requests = [objectRecognition]
        } catch { print("Error: Could not set up Vision") }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try imageRequestHandler.perform(self.requests)
        } catch { print(error) }
    }

    //Output video and classifications
    func updateClassificationLabel(_ results: [Any]) {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        for observation in results {
            guard let observation = observation as? VNClassificationObservation else { return }
            prediction == nil ? prediction = observation : nil
            observation.confidence >= prediction!.confidence ? prediction = observation : nil
            textOverlay.string = String(format: "\(prediction!.identifier) : %.2f", prediction!.confidence)
        }
        CATransaction.commit()
        prediction = nil
    }
}
