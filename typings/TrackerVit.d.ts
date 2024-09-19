import { Mat } from './Mat.d';
import { Rect } from './Rect.d';

export class TrackerVit {
  /**
   * Creates a new TrackerVit object.
   * @param netModelPath Optional path to the Vit Tracker net ONNX model.
   */
  constructor(netModelPath?: string);
  
  clear(): void;
  init(frame: Mat, boundingBox: Rect): boolean;
  update(frame: Mat): Rect;
}
