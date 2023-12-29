import sweetify
from django.shortcuts import redirect, render
from django.views.generic.base import TemplateView
from django.core.files.storage import FileSystemStorage
from BrainApp.utils import inference, extract_spectrogram, load_model

# Global Variable
model = load_model()

# Create your views here.
class IndexView(TemplateView):
    template_name = 'index.html'

    def get_context_data(self, **kwargs):
        return {}


class PredictView(TemplateView):
    template_name = 'predict.html'

    def post(self, *args, **kwargs):
        myfile = self.request.FILES['file']
        extension = str(myfile).split('.')[-1]
        allowed_extension = ['edf', 'png', 'jpeg', 'jpg']
        if extension not in allowed_extension:
            sweetify.error(request=self.request, title='Format file tidak didukung')
            return redirect('/predict')

        fs = FileSystemStorage(location='upload')
        filename = fs.save(content=myfile, name=str(myfile))
        if extension in allowed_extension[1:]:
            result_class = inference(model=model, file_path=filename, image=True)
        else:
            result_class = inference(model=model, file_path=filename, image=False)

        ctx = {
            'hasil': result_class
        }

        return render(request=self.request, template_name='result.html', context=ctx)

    def get_context_data(self, **kwargs):
        return {}


class ResultView(TemplateView):
    template_name = 'result.html'

    def get_context_data(self, **kwargs):
        return {}