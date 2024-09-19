import { expect } from 'chai';
import { Mat, TrackerVit } from '../../../typings';
import { getTestContext } from '../model';
import toTest from '../toTest';
import path from 'path';  // Import path module to handle file paths
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const netModelPath = path.join(__dirname, '/TrackerVitModel/vittracknet.onnx');
let tracker: TrackerVit

if (toTest.tracking) {
  const {
    cv,
    cvVersionGreaterEqual,
    getTestImg,
  } = getTestContext();

  const hasVit = cvVersionGreaterEqual(4, 9, 0);

  (hasVit ? describe : describe.skip)('TrackerVit', () => {
    let testImg: Mat;

    before(() => {
      testImg = getTestImg();
    });

    describe('constructor', () => {
      it('can be constructed', () => {
        tracker = new cv.TrackerVit(netModelPath);
        expect(tracker).to.have.property('init').to.be.a('function');
        expect(tracker).to.have.property('update').to.be.a('function');
      });
    });

    describe('init', () => {
      it('should throw if no args', () => {
        // @ts-expect-error missing args
        expect(() => tracker.init()).to.throw('TrackerVit::Init - Error: expected argument 0 to be of type');
      });

      it('can be called with frame and initial box', () => {
        const ret = tracker.init(testImg, new cv.Rect(0, 0, 10, 10));
        expect(ret).to.be.true;
      });
    });

    describe('update', () => {
      it('should throw if no args', () => {
        // @ts-expect-error missing args
        expect(() => tracker.update()).to.throw('TrackerVit::Update - Error: expected argument 0 to be of type');
      });

      it('returns bounding box', () => {
        tracker.init(testImg, new cv.Rect(0, 0, 10, 10));
        const rect = tracker.update(testImg);
        expect(rect).to.be.instanceOf(cv.Rect);
      });
    });

  });
}
