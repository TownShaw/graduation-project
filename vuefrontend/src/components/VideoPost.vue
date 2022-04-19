<template>
  <div id="inputFiles">
    <p>Post Data to Backend:</p>
    <input type="file" name="video" @change="handleFileUpLoad($event, 'video')">
    <input type="file" name="subtitle" @change="handleFileUpLoad($event, 'subtitle')">
    <input type="submit" name="submit" @click="submitFiles()">
  </div>
</template>

<script>
import { postRequest } from '../utils/api'

export default {
  name: "VideoPost",
  emits: ["resultChange"],
  data() {
    return {
      subtitle: undefined,
      video: undefined
    }
  },
  methods: {
    handleFileUpLoad(event, fileKind) {
      if (fileKind === 'video') {
        this.video = event.target.files[0]
      } else {
        this.subtitle = event.target.files[0]
      }
    },
    submitFiles() {
      const formData = new FormData();
      formData.append('video', this.video);
      formData.append('subtitle', this.subtitle);
      let headers = {
            // 'Origin':'http://localhost:4000',
            "Content-Type": "multipart/form-data"
            // "Content-Type": "application/json"
        }
      // postRequest('', {"video": this.video, "subtitle": this.subtitle}).then((res) => {
      postRequest('', formData, headers).then((resp) => {
        if (resp.status == 200) {
          let jsonData = resp.data;   // predict results
          this.$emit("resultChange", jsonData);  // emit results to parent component
        }
      });
    }
  },
}
</script>

<style>

</style>
