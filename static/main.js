(function() {
    var uploader = new plupload.Uploader({
      runtimes : 'html5,flash,silverlight,html4',
      browse_button : 'pickfiles', // you can pass in id...
      container: document.getElementById('container'), 
      url : "/",
      filters : {
        max_file_size : '10mb',
        mime_types: [
          {title : "Image files", extensions : "jpg,gif,png"},
          {title : "Zip files", extensions : "zip"}
        ]
      },
   
      // Flash settings
      flash_swf_url : 'static/plupload/js/Moxie.swf',
   
      // Silverlight settings
      silverlight_xap_url : 'static/plupload/js/Moxie.xap',
   
      init: {
        PostInit: function() {
          document.getElementById('filelist').innerHTML = '';
          document.getElementById('uploadfiles').onclick = function() {
            uploader.start();
            return false;
          };
        },
   
        FilesAdded: function(up, files) {
          plupload.each(files, function(file) {
            document.getElementById('filelist').innerHTML += '<div id="' + file.id 
              + '">' + file.name + ' (' + plupload.formatSize(file.size) + ') <b></b></div>';
          });
        },
   
        UploadProgress: function(up, file) {
          document.getElementById(file.id).getElementsByTagName('b')[0].innerHTML = '<span>' + file.percent + "%</span>";
        },
   
        Error: function(up, err) {
          document.getElementById('console').innerHTML += "\nError #" + err.code + ": " + err.message;
        }
      }
    });
     
    uploader.init();
}).call(this);
