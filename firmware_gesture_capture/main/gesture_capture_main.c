/*
 * Gesture capture firmware for the ESP32-S3.
 *
 * Streams 4 capacitive touch channels over serial so collect_gestures.py can
 * record labeled examples. Pads are GPIO 7, 8, 9, 12, in that physical order
 * along one header row. I checked all 14 touch channels on the board first;
 * GPIO5 does not respond on my unit, so it is skipped.
 *
 * Uses driver/touch_sens.h in polling mode, which avoids the S3 touch
 * interrupt-flag erratum. The int8 config values for touch V2 are inlined so
 * the project does not depend on the IDF example components.
 *
 * Serial format (115200 baud, shows up as /dev/ttyACM0):
 *   #GCAP ch=7,8,9,12 fs=50            banner, once at boot
 *   D <ms> <d7> <d8> <d9> <d12>        one per sample; d = smooth - benchmark
 * d is how far the reading sits above its resting baseline. A finger raises it
 * by hundreds to thousands; the noise floor is about +/-5.
 */

#include <stdio.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/touch_sens.h"
#include "esp_timer.h"
#include "esp_check.h"

#define GCAP_NUM_CH     4
#define GCAP_SAMPLE_HZ  50
#define GCAP_PERIOD_MS  (1000 / GCAP_SAMPLE_HZ)

static const int s_channel_id[GCAP_NUM_CH] = { 7, 8, 9, 12 };

static touch_sensor_handle_t  s_sens = NULL;
static touch_channel_handle_t s_chan[GCAP_NUM_CH];

void app_main(void)
{
    /* Controller with the ESP32-S3 (touch V2) default sample configuration. */
    touch_sensor_sample_config_t sample_cfg[TOUCH_SAMPLE_CFG_NUM] = {
        TOUCH_SENSOR_V2_DEFAULT_SAMPLE_CONFIG(500, TOUCH_VOLT_LIM_L_0V5, TOUCH_VOLT_LIM_H_2V2),
    };
    touch_sensor_config_t sens_cfg = TOUCH_SENSOR_DEFAULT_BASIC_CONFIG(TOUCH_SAMPLE_CFG_NUM, sample_cfg);
    ESP_ERROR_CHECK(touch_sensor_new_controller(&sens_cfg, &s_sens));

    /* Allocate the 4 touch channels (relative-threshold config for V2). */
    touch_channel_config_t chan_cfg = {
        .active_thresh = { 2000 },
        .charge_speed = TOUCH_CHARGE_SPEED_7,
        .init_charge_volt = TOUCH_INIT_CHARGE_VOLT_DEFAULT,
    };
    for (int i = 0; i < GCAP_NUM_CH; i++) {
        ESP_ERROR_CHECK(touch_sensor_new_channel(s_sens, s_channel_id[i], &chan_cfg, &s_chan[i]));
    }

    /* Default hardware filter (smooths the raw reading). */
    touch_sensor_filter_config_t filter_cfg = TOUCH_SENSOR_DEFAULT_FILTER_CONFIG();
    ESP_ERROR_CHECK(touch_sensor_config_filter(s_sens, &filter_cfg));

    /* Initial scanning so the per-channel benchmark/baseline stabilises. */
    ESP_ERROR_CHECK(touch_sensor_enable(s_sens));
    for (int i = 0; i < 3; i++) {
        ESP_ERROR_CHECK(touch_sensor_trigger_oneshot_scanning(s_sens, 2000));
    }
    ESP_ERROR_CHECK(touch_sensor_disable(s_sens));

    /* Enable + start continuous scanning; we poll the results. */
    ESP_ERROR_CHECK(touch_sensor_enable(s_sens));
    ESP_ERROR_CHECK(touch_sensor_start_continuous_scanning(s_sens));

    printf("#GCAP ch=7,8,9,12 fs=%d\n", GCAP_SAMPLE_HZ);

    TickType_t last_wake = xTaskGetTickCount();
    while (1) {
        int32_t delta[GCAP_NUM_CH];
        for (int i = 0; i < GCAP_NUM_CH; i++) {
            uint32_t smooth[TOUCH_SAMPLE_CFG_NUM] = {0};
            uint32_t bench[TOUCH_SAMPLE_CFG_NUM]  = {0};
            if (touch_channel_read_data(s_chan[i], TOUCH_CHAN_DATA_TYPE_SMOOTH, smooth) != ESP_OK ||
                touch_channel_read_data(s_chan[i], TOUCH_CHAN_DATA_TYPE_BENCHMARK, bench) != ESP_OK) {
                delta[i] = 0;
            } else {
                delta[i] = (int32_t)smooth[0] - (int32_t)bench[0];
            }
        }
        uint32_t ms = (uint32_t)(esp_timer_get_time() / 1000);
        printf("D %" PRIu32 " %" PRId32 " %" PRId32 " %" PRId32 " %" PRId32 "\n",
               ms, delta[0], delta[1], delta[2], delta[3]);

        /* Fixed-rate sampling so host-side timestamps aren't required. */
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(GCAP_PERIOD_MS));
    }
}
